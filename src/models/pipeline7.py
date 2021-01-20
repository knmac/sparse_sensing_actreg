"""Pipeline7 - Full pipeline with all components

Similar to pipeline6, but trainable and allow batch size != 1
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader


class Pipeline7(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, attention_layer, attention_dim, dropout,
                 low_feat_model_cfg, high_feat_model_cfg, hallu_model_cfg,
                 actreg_model_cfg, spatial_sampler_cfg, temporal_sampler_cfg,
                 hallu_pretrained_weights, actreg_pretrained_weights,
                 feat_process_type, freeze_hallu, freeze_actreg):
        super(Pipeline7, self).__init__(device)

        # Turn off cudnn benchmark because of different input size
        # torch.backends.cudnn.benchmark = False

        # Only support layer3_0 for now
        # Affecting the implementation of 1st and 2nd half splitting
        assert attention_layer == ['layer3', '0']

        # Save the input arguments
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.feat_process_type = feat_process_type  # [add, cat]

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

        # Generate temporal sampler
        name, params = ConfigLoader.load_model_cfg(temporal_sampler_cfg)
        params.update({
            'attention_dim': self.attention_dim,
        })
        self.temporal_sampler = model_factory.generate(name, **params)

        # Generate spatial sampler
        name, params = ConfigLoader.load_model_cfg(spatial_sampler_cfg)
        self.spatial_sampler = model_factory.generate(name, **params)

        # Feature processing functions
        if self.feat_process_type == 'add':
            # Combine the top k features from high rgb by adding,
            # Make sure the feature dimensions are the same
            assert self.low_feat_model.feature_dim == self.high_feat_model.feature_dim, \
                'Feature dimensions must be the same to add'
            real_dim = self.low_feat_model.feature_dim
        elif self.feat_process_type == 'cat':
            real_dim = self.low_feat_model.feature_dim * len(modality) + \
                self.high_feat_model.feature_dim * self.spatial_sampler.top_k
        else:
            raise NotImplementedError

        # Generate hallucination model
        name, params = ConfigLoader.load_model_cfg(hallu_model_cfg)
        assert name in ['HalluConvLSTM2']
        params.update({
            'attention_dim': self.attention_dim,
        })
        self.hallu_model = model_factory.generate(name, device=device, **params)
        self.hallu_model.load_model(hallu_pretrained_weights)

        if freeze_hallu:
            for param in self.hallu_model.parameters():
                param.requires_grad_(False)

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
        self.actreg_model.load_model(actreg_pretrained_weights)

        if freeze_actreg:
            for param in self.actreg_model.parameters():
                param.requires_grad_(False)

    def _check_freeze(self, model):
        lst = []
        for param in model.parameters():
            lst.append(param.requires_grad)
        print('Required grads: {}/{}'.format(sum(lst), len(lst)))

    def forward(self, x, avg=True, get_extra=False):
        """Forwad a sequence of frame
        """
        rgb_high = x['RGB']
        rgb_low = rgb_high[:, :, ::2, ::2]
        spec = x['Spec']
        batch_size = rgb_high.shape[0]
        # assert batch_size == 1, 'Only support batch_size 1 for now'

        # (B, T*C, H, W) -> (B, T, C, H, W)
        rgb_low = rgb_low.view((-1, self.num_segments, 3) + rgb_low.size()[-2:])
        rgb_high = rgb_high.view((-1, self.num_segments, 3) + rgb_high.size()[-2:])
        spec = spec.view((-1, self.num_segments, 1) + spec.size()[-2:])

        # =====================================================================
        # Prescan and hallucinate by batch because this is conducted regardless
        # of skipping
        # =====================================================================
        # First half of feature low rgb feat --> (B, T, C, H, W)
        prescan_feat = self.first_half_forward(
            rgb_low.view((-1, 3) + rgb_low.size()[-2:]), self.low_feat_model.rgb
        )
        prescan_feat = prescan_feat.view((-1, self.num_segments) + prescan_feat.size()[-3:])

        # Get attention of all frames --> (B, T, C, H, W)
        attn = self.low_feat_model.rgb.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        assert attn is not None, 'Fail to retrieve attention'
        attn = attn.view((-1, self.num_segments) + attn.size()[-3:])

        # =====================================================================
        # Process each sample separately because skipping is non-uniform across
        # batch domain
        # =====================================================================
        outputs, extra_outputs = [], []
        for i in range(batch_size):
            # Reset samplers at the start of the sequence
            self.temporal_sampler.reset()
            self.spatial_sampler.reset()

            # Forward frame by frame
            actreg_mem, hallu_mem, pred, hallu = None, None, None, None
            all_skip, all_ssim, all_time = [], [], []
            all_pred_verb, all_pred_noun = [], []
            for t in range(self.num_segments):
                st = time.time()
                (pred, hallu,
                 actreg_mem,
                 hallu_mem,
                 skipped,
                 ssim) = self.forward_frame(
                    prescan_feat[i, t].unsqueeze(dim=0), attn[i, t].unsqueeze(dim=0),
                    spec[i, t].unsqueeze(dim=0), rgb_high[i, t].unsqueeze(dim=0),
                    old_pred=pred, old_hallu=hallu,
                    actreg_mem=actreg_mem, hallu_mem=hallu_mem,
                )

                # Collect results
                all_time.append(time.time()-st)
                all_skip.append(skipped)
                all_ssim.append(ssim)
                all_pred_verb.append(pred[0].unsqueeze(dim=1))
                all_pred_noun.append(pred[1].unsqueeze(dim=1))

            # Prepare outputs
            all_pred_verb = torch.cat(all_pred_verb, dim=1)
            all_pred_noun = torch.cat(all_pred_noun, dim=1)
            if avg:
                output = (all_pred_verb.mean(dim=1), all_pred_noun.mean(dim=1))
            else:
                output = (all_pred_verb, all_pred_noun)

            extra_output = {
                'skip': all_skip,
                'time': all_time,
                'ssim': all_ssim,
            }

            # Collect batch
            outputs.append(output)
            extra_outputs.append(extra_output)

        # Manipulate batch and output fields
        out_verb = torch.cat([x[0] for x in outputs], dim=0)
        out_noun = torch.cat([x[1] for x in outputs], dim=0)
        outputs = (out_verb, out_noun)

        if get_extra:
            return outputs, extra_outputs
        return outputs

    def forward_frame(self, prescan_feat, attn, spec, rgb_high,
                      old_pred=None, old_hallu=None,
                      actreg_mem=None, hallu_mem=None):
        """Forward a single frame

        Args:
            prescan_feat: feature from the first half of the network, using low
                resolution input
            attn: attention wrt the prescan features
            spec: spectrogram input of time t
            rgb_high: high resolution rgb input of time t
            old_pred: label prediction of time t-1
            old_hallu: hallucination of attention for frame t (from time t-1)
            actreg_mem: memory of actreg_model from time t-1
            hallu_mem: memory of hallu_model from time t-1

        Return:
            pred: label prediction of time t
            hallu: hallucination of attention for time t+1 (from time t)
            actreg_mem: updated memory of actreg_model
            hallu_mem: updated memory of hallu_model
            skipped: whether frame t is skipped
            ssim: ssim value between attention and hallucination
        """
        batch_size = rgb_high.shape[0]
        assert batch_size == 1, 'Only support batch_size 1 when forward a frame'

        # =====================================================================
        # Prescan and hallucinate
        # =====================================================================
        # First half of feature low rgb feat
        # prescan_feat = self.first_half_forward(rgb_low, self.low_feat_model.rgb)

        # Get attention of current frame --> (B, C, H, W)
        # attn = self.low_feat_model.rgb.get_attention_weight(
        #     l_name=self.attention_layer[0],
        #     m_name=self.attention_layer[1],
        #     aggregated=True,
        # )
        # assert attn is not None, 'Fail to retrieve attention'

        # Hallucinate attention of time t+1, using real attention of time t
        hallu, hallu_mem = self.hallu_model(attn.unsqueeze(dim=1), hallu_mem)
        assert hallu.shape[1] == 1
        hallu = hallu[:, 0]

        # Temporal sampler
        to_skip, ssim = self.temporal_sampler.sample_frame(attn, old_hallu)

        # =====================================================================
        # Case 1: skipping the current frame
        # =====================================================================
        if to_skip:
            return (old_pred,    # Don't update prediction
                    hallu,       # New hallucination
                    actreg_mem,  # Don't update actreg_mem
                    hallu_mem,   # New hallu_mem
                    to_skip,
                    ssim,
                    )

        # =====================================================================
        # Case 2: not skipping the current frame
        # =====================================================================
        # Second half of RGB low
        low_feat = self.second_half_forward(prescan_feat, self.low_feat_model.rgb)

        # Spec
        spec_feat = self.low_feat_model.spec(spec)

        # Spatial sampler -----------------------------------------------------
        # Compute bboxes -> (B, top_k, 4)
        bboxes = self.spatial_sampler.sample_frame(attn, rgb_high.shape[-1],
                                                   reorder=True)

        # Extract regions and feed in high_feat_model
        high_feat = []
        for k in range(self.spatial_sampler.top_k):
            top = bboxes[0, k, 0]
            left = bboxes[0, k, 1]
            bottom = bboxes[0, k, 2]
            right = bboxes[0, k, 3]

            region = rgb_high[:, :, top:bottom, left:right]
            high_feat_k = self.high_feat_model({'RGB': region})
            high_feat.append(high_feat_k)

        # Action recognition --------------------------------------------------
        if self.feat_process_type == 'add':
            all_feats = low_feat + spec_feat
            for k in range(self.spatial_sampler.top_k):
                all_feats += high_feat[k]
        elif self.feat_process_type == 'cat':
            all_feats = torch.cat([low_feat, spec_feat] + high_feat, dim=1)

        assert all_feats.ndim == 2
        all_feats = all_feats.unsqueeze(dim=1)

        pred, actreg_mem = self.actreg_model(all_feats, actreg_mem)

        return pred, hallu, actreg_mem, hallu_mem, to_skip, ssim

    def first_half_forward(self, x, san):
        """Warper of the forward function of SAN to run the first half, up to
        layer3_0
        """
        x = san.relu(san.bn_in(san.conv_in(x)))
        x = san.relu(san.bn0(san.layer0(san.conv0(san.pool(x)))))
        x = san.relu(san.bn1(san.layer1(san.conv1(san.pool(x)))))
        x = san.relu(san.bn2(san.layer2(san.conv2(san.pool(x)))))

        # Run layer3_0
        x = san.layer3[0](san.conv3(san.pool(x)))
        return x

    def second_half_forward(self, x, san):
        """Warper of the forward function of SAN to run the second half, from
        after layer3_0 (starting with layer3_1)
        """
        # Run from layer3_1 to the end of layer3
        # x = san.relu(san.bn3(san.layer3(san.conv3(san.pool(x)))))
        x = san.relu(san.bn3(san.layer3[1:](x)))

        x = san.relu(san.bn4(san.layer4(san.conv4(san.pool(x)))))
        x = san.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def freeze_fn(self, freeze_mode):
        self.low_feat_model.freeze_fn(freeze_mode)
        self.high_feat_model.freeze_fn(freeze_mode)
        # self.hallu_model.freeze_fn(freeze_mode)
        # self.actreg_model.freeze_fn(freeze_mode)

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
