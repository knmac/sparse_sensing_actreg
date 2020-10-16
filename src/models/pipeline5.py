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

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader


class Pipeline5(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim, dropout,
                 high_feat_model_cfg, low_feat_model_cfg, spatial_sampler_cfg,
                 actreg_model_cfg, reduce_dim_low, reduce_dim_high):
        super(Pipeline5, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.reduce_dim_low = reduce_dim_low  # Applied for rgb_low+spec together
        self.reduce_dim_high = reduce_dim_high  # Applied for each rgh_high of top_k

        # Generate feature extraction models for low resolutions
        name, params = ConfigLoader.load_model_cfg(low_feat_model_cfg)
        assert params['new_input_size'] == 112, \
            'Only support low resolutions of 112 for now'
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
        })
        self.low_feat_model = model_factory.generate(name, device=device, **params)

        # Generate feature extraction models for high resolutions
        name, params = ConfigLoader.load_model_cfg(high_feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': ['RGB'],  # Remove spec because low_model already has it
        })
        self.high_feat_model = model_factory.generate(name, device=device, **params)

        # Generate spatial sampler
        name, params = ConfigLoader.load_model_cfg(spatial_sampler_cfg)
        self.spatial_sampler = model_factory.generate(name, **params)

        # FC layers to reduce visual feature dimension
        self.fc_reduce_low = nn.Linear(
            in_features=self.low_feat_model.feature_dim*len(modality),
            out_features=reduce_dim_low,
        ).to(device)
        self.fc_reduce_high = nn.Linear(
            in_features=self.high_feat_model.feature_dim,
            out_features=reduce_dim_high,
        ).to(device)
        self.relu = nn.ReLU(inplace=True)

        # Generate action recognition model
        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        assert name in ['ActregGRU2'], \
            'Unsupported model: {}'.format(name)
        real_dim = self.fc_reduce_low.out_features + \
            self.fc_reduce_high.out_features*self.spatial_sampler.top_k
        params.update({
            'feature_dim': 0,  # Use `real_dim` instead
            'extra_dim': real_dim,
            'modality': self.modality,
            'num_class': self.num_class,
            'dropout': self.dropout,
        })
        self.actreg_model = model_factory.generate(name, device=device, **params)

    def forward(self, x):
        _rgb_high = x['RGB']
        _rgb_low = _rgb_high[:, :, ::2, ::2]  # [B, T*C, H, W]
        _spec = x['Spec']
        batch_size = _rgb_high.shape[0]

        # Extract low resolutions features and attention ----------------------
        # Extract features
        low_feat = self.low_feat_model({'RGB': _rgb_low, 'Spec': _spec})
        low_feat = low_feat.view([batch_size,
                                  self.num_segments,
                                  len(self.modality)*self.low_feat_model.feature_dim])
        low_feat = self.relu(self.fc_reduce_low(low_feat))

        # Extract attention
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

        # Extract regions and feed in high_feat_model -------------------------
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

            # Concat acrose batch dim and collect
            high_feat_k = torch.cat(high_feat_k, dim=0)
            high_feat.append(self.relu(self.fc_reduce_high(high_feat_k)))

        assert len(high_feat) == self.spatial_sampler.top_k
        assert high_feat[0].shape[0] == batch_size
        assert high_feat[0].shape[1] == self.num_segments

        # Action recognition --------------------------------------------------
        all_feats = torch.cat([low_feat] + high_feat, dim=2)
        assert all_feats.ndim == 3

        target_dim = self.reduce_dim_low + self.reduce_dim_high*self.spatial_sampler.top_k
        assert all_feats.shape[2] == target_dim

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
