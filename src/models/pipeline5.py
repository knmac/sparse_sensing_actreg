"""Pipeline version 5

Has spatial sampler.

Train only action recognition (others are frozen) but the inputs are from
multiple resolutions
"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader


class Pipeline5(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim, dropout,
                 high_feat_model_cfg, low_feat_model_cfg, spatial_sampler_cfg,
                 actreg_model_cfg, feat_process_type, using_cupy, reduce_dim=None,
                 ignore_lowres=False):
        super(Pipeline5, self).__init__(device)

        # Turn off cudnn benchmark because of different input size
        # This is only effective whenever pipeline5 is used
        # torch.backends.cudnn.benchmark = False

        # Save the input arguments
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.feat_process_type = feat_process_type  # [reduce, add, cat]
        self.using_cupy = using_cupy
        self.ignore_lowres = ignore_lowres  # ignore low res feature and use only sampled high res
        if ignore_lowres:
            assert feat_process_type == 'cat'  # only works with cat

        # Generate feature extraction models for low resolutions
        name, params = ConfigLoader.load_model_cfg(low_feat_model_cfg)
        assert params['new_input_size'] in [112, 64], \
            'Only support low resolutions of 112 or 64 for now'
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
            'using_cupy': self.using_cupy,
        })
        self.low_feat_model = model_factory.generate(name, device=device, **params)

        # Generate feature extraction models for high resolutions
        name, params = ConfigLoader.load_model_cfg(high_feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': ['RGB'],  # Remove spec because low_model already has it
            'using_cupy': self.using_cupy,
        })
        self.high_feat_model = model_factory.generate(name, device=device, **params)

        # Generate spatial sampler
        name, params = ConfigLoader.load_model_cfg(spatial_sampler_cfg)
        self.spatial_sampler = model_factory.generate(name, **params)

        # Feature processing functions
        if self.feat_process_type == 'reduce':
            # Reduce dimension of each feature
            self.reduce_dim = reduce_dim

            # FC layers to reduce feature dimension
            self.fc_reduce_low = nn.Linear(
                in_features=self.low_feat_model.feature_dim,
                out_features=reduce_dim,
            ).to(device)
            self.fc_reduce_high = nn.Linear(
                in_features=self.high_feat_model.feature_dim,
                out_features=reduce_dim,
            ).to(device)
            self.fc_reduce_spec = nn.Linear(
                in_features=self.low_feat_model.feature_dim,
                out_features=reduce_dim,
            ).to(device)
            self.relu = nn.ReLU(inplace=True)

            real_dim = self.fc_reduce_low.out_features + \
                self.fc_reduce_spec.out_features + \
                self.fc_reduce_high.out_features*self.spatial_sampler.top_k
        elif self.feat_process_type == 'add':
            # Combine the top k features from high rgb by adding,
            # Make sure the feature dimensions are the same
            assert self.low_feat_model.feature_dim == self.high_feat_model.feature_dim, \
                'Feature dimensions must be the same to add'
            real_dim = self.low_feat_model.feature_dim
        elif self.feat_process_type == 'cat':
            if self.ignore_lowres:
                real_dim = self.low_feat_model.feature_dim * (len(modality)-1) + \
                    self.high_feat_model.feature_dim * self.spatial_sampler.top_k
            else:
                real_dim = self.low_feat_model.feature_dim * len(modality) + \
                    self.high_feat_model.feature_dim * self.spatial_sampler.top_k
        else:
            raise NotImplementedError

        # Generate action recognition model
        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        assert name in ['ActregGRU2', 'ActregFc'], \
            'Unsupported model: {}'.format(name)
        if name == 'ActregGRU2':
            params.update({
                'feature_dim': 0,  # Use `real_dim` instead
                'extra_dim': real_dim,
                'modality': self.modality,
                'num_class': self.num_class,
                'dropout': self.dropout,
            })
        elif name == 'ActregFc':
            params.update({
                'feature_dim': real_dim,
                'modality': self.modality,
                'num_class': self.num_class,
                'dropout': self.dropout,
                'num_segments': self.num_segments,
            })
        self.actreg_model = model_factory.generate(name, device=device, **params)

    def _downsample(self, x):
        """Downsample/rescale high resolution image to make low resolution version

        Args:
            x: high resolution image tensor, shape of (B, T*3, H, W)

        Return:
            Low resolution version of x
        """
        high_dim = self.high_feat_model.input_size['RGB']
        low_dim = self.low_feat_model.input_size['RGB']
        down_factor = high_dim / low_dim

        if isinstance(down_factor, int):
            return x[:, :, ::down_factor, ::down_factor]
        return F.interpolate(x, size=low_dim, mode='bilinear', align_corners=False)

    def forward(self, x):
        _rgb_high = x['RGB']
        _rgb_low = self._downsample(_rgb_high)
        _spec = x['Spec']
        batch_size = _rgb_high.shape[0]

        # Extract low resolutions features ------------------------------------
        assert self.low_feat_model.modality == ['RGB', 'Spec']
        low_feat, spec_feat = self.low_feat_model({'RGB': _rgb_low, 'Spec': _spec},
                                                  return_concat=False)

        # (B*T, C) --> (B, T, C)
        low_feat = low_feat.view([batch_size,
                                  self.num_segments,
                                  self.low_feat_model.feature_dim])
        spec_feat = spec_feat.view([batch_size,
                                    self.num_segments,
                                    self.low_feat_model.feature_dim])

        # Feature processing
        if self.feat_process_type == 'reduce':
            low_feat = self.relu(self.fc_reduce_low(low_feat))
            spec_feat = self.relu(self.fc_reduce_spec(spec_feat))
        elif self.feat_process_type == 'add':
            # Do nothing
            pass
        elif self.feat_process_type == 'cat':
            # Do nothing
            pass

        # Retrieve attention --------------------------------------------------
        attn = self.low_feat_model.rgb.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        attn = attn.view([-1, self.num_segments] + list(attn.shape[1:]))
        self._attn = attn

        # Spatial sampler -----------------------------------------------------
        # Compute bboxes -> (B, T, top_k, 4)
        bboxes = self.spatial_sampler.sample_multiple_frames(
            attn, _rgb_high.shape[-1], reorder=True, avg_across_time=True)

        # (B, T*C, H, W) -> (B, T, C, H, W)
        _rgb_high = _rgb_high.view((-1, self.num_segments, 3) + _rgb_high.size()[-2:])
        # self._check(_rgb_high, attn, bboxes)

        # Extract regions and feed in high_feat_model
        high_feat = []
        for k in range(self.spatial_sampler.top_k):
            high_feat_k = []
            for b in range(batch_size):
                tops = bboxes[b, :, k, 0]
                lefts = bboxes[b, :, k, 1]
                bottoms = bboxes[b, :, k, 2]
                rights = bboxes[b, :, k, 3]

                # Batch regions across time b/c of consisting size
                regions_k_b = []
                for t in range(self.num_segments):
                    regions_k_b.append(
                        _rgb_high[b, t, :,
                                  tops[t]:bottoms[t],
                                  lefts[t]:rights[t]
                                  ].unsqueeze(dim=0))
                regions_k_b = torch.cat(regions_k_b, dim=0)

                # Tensor manipulation to prepare
                regions_k_b = regions_k_b.unsqueeze(dim=0)
                regions_k_b = regions_k_b.view(
                    [1, regions_k_b.shape[1]*regions_k_b.shape[2],
                     regions_k_b.shape[3], regions_k_b.shape[4]])

                # Feed the regions to high_feat_model
                out = self.high_feat_model({'RGB': regions_k_b})
                high_feat_k.append(out.unsqueeze(dim=0))

            # Concat across batch dim and collect
            high_feat_k = torch.cat(high_feat_k, dim=0)
            if self.feat_process_type == 'reduce':
                high_feat.append(self.relu(self.fc_reduce_high(high_feat_k)))
            elif self.feat_process_type == 'add':
                high_feat.append(high_feat_k)
            elif self.feat_process_type == 'cat':
                high_feat.append(high_feat_k)

        assert len(high_feat) == self.spatial_sampler.top_k
        assert high_feat[0].shape[0] == batch_size
        assert high_feat[0].shape[1] == self.num_segments

        # Action recognition --------------------------------------------------
        if self.feat_process_type == 'reduce':
            all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=2)
        elif self.feat_process_type == 'add':
            all_feats = low_feat + spec_feat
            for k in range(self.spatial_sampler.top_k):
                all_feats += high_feat[k]
        elif self.feat_process_type == 'cat':
            if self.ignore_lowres:
                all_feats = torch.cat([spec_feat] + high_feat, dim=2)
            else:
                all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=2)

        assert all_feats.ndim == 3

        output, hidden = self.actreg_model(all_feats, hidden=None)

        # Does not need belief_loss because the function compare_belief is not
        # available here
        return output

    def _check(self, img, attn, bboxes, ix=0):
        """Visualize to check the results of spatial sampler. For debugging only
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid', {'axes.grid': False})

        img = img[ix].cpu().detach().numpy()
        attn = attn[ix].cpu().detach().numpy()
        bboxes = bboxes[ix]

        img = np.transpose(img, [0, 2, 3, 1]) + np.array([104, 117, 128])
        img = img[..., ::-1].astype(np.uint8)
        attn = attn.mean(axis=1)

        fig, axes = plt.subplots(3, self.num_segments)
        for t in range(self.num_segments):
            axes[0, t].imshow(img[t])
            axes[1, t].imshow(attn[t], vmin=attn.min(), vmax=attn.max())

            frame = np.zeros(img.shape[1:], dtype=np.uint8)
            bbox = bboxes[t]
            for k in range(3):
                frame[bbox[k, 0]:bbox[k, 2], bbox[k, 1]:bbox[k, 3], k] = 255
            axes[2, t].imshow(frame)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def freeze_fn(self, freeze_mode):
        self.low_feat_model.freeze_fn(freeze_mode)
        self.high_feat_model.freeze_fn(freeze_mode)

    @property
    def input_mean(self):
        return self.high_feat_model.input_mean

    @property
    def input_std(self):
        # because low_feat_model has spec std
        return self.low_feat_model.input_std

    @property
    def crop_size(self):
        return self.high_feat_model.input_size

    @property
    def scale_size(self):
        return self.high_feat_model.scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
