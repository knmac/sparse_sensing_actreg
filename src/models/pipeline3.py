"""Pipeline version 3 - Run only hallucinator
"""
import sys
import os

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from .base_model import BaseModel
from .pytorch_ssim.ssim import SSIM
from src.utils.load_cfg import ConfigLoader


class Pipeline3(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, rnn_win_len, attention_layer, attention_dim,
                 feat_model_cfg, hallu_model_cfg):
        super(Pipeline3, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.rnn_win_len = rnn_win_len  # N frames to feed in RNN at a time
        self.attention_layer = attention_layer
        self.attention_dim = attention_dim

        # Generate feature extraction model
        name, params = ConfigLoader.load_model_cfg(feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
        })
        self.feat_model = model_factory.generate(name, device=device, **params)

        # Generate hallucination model
        name, params = ConfigLoader.load_model_cfg(hallu_model_cfg)
        assert name in ['HalluConvLSTM2'], \
            'Unsupported model: {}'.format(name)
        params.update({
            'attention_dim': self.attention_dim,
        })
        self.hallu_model = model_factory.generate(name, device=device, **params)

        # Loss for belief propagation
        self.belief_criterion = SSIM(window_size=3, channel=self.attention_dim[0])

    def forward(self, x):
        # Extract features ----------------------------------------------------
        self.feat_model(x)  # Only feed forward to get the attention

        # Retrieve attention from feature model
        attn = self.feat_model.rgb.get_attention_weight(
            l_name=self.attention_layer[0],
            m_name=self.attention_layer[1],
            aggregated=True,
        )
        attn = attn.view([-1, self.num_segments] + list(attn.shape[1:]))
        self._attn = attn

        # Hallucination -------------------------------------------------------
        # Attention shape: [B, T, C, H, W]
        patch_in = attn[:, :self.rnn_win_len]  # first patch
        hallu = torch.zeros_like(attn)
        hidden = None
        first_patch = True

        idx = self.rnn_win_len - 1
        while idx < self.num_segments:
            # Feed the patch to RNN
            patch_out, hidden = self.hallu_model(patch_in, hidden)
            last_out = patch_out[:, -1].unsqueeze(dim=1)

            # Collect the output hallucination
            if first_patch:
                hallu[:, :self.rnn_win_len] = patch_out
                first_patch = False
            else:
                hallu[:, idx] = last_out

            # Update the new output to patch_in
            patch_in = torch.cat([patch_in[:, 1:], last_out], dim=1)
            idx += 1

        # hallu = self.hallu_model(attn)
        self._hallu = hallu

        # Dummy classification output -----------------------------------------
        # There will be no gradient for the action recognition part. This is
        # only here to make train_val.py happy
        batch_size = attn.shape[0]
        output = (
            torch.zeros([batch_size, self.num_class[0]]).to(self.device),
            torch.zeros([batch_size, self.num_class[1]]).to(self.device),
        )

        return output, self.compare_belief().unsqueeze(dim=0)

    def compare_belief(self):
        """Compare between attention and hallucination. Do NOT call directly.

        If using multiple GPUs, self._hallu and self._attn will not be available
        """
        assert hasattr(self, '_attn') and hasattr(self, '_hallu'), \
            'Attributes are not found'
        assert self._attn.shape == self._hallu.shape, 'Mismatching shapes'
        assert torch.all(self._attn >= 0) and torch.all(self._hallu >= 0)

        # Get attention of future frames
        attn_future = self._attn[:, self.rnn_win_len:]

        # Get hallucination from current frames (for future frames)
        hallu_current = self._hallu[:, self.rnn_win_len-1:-1]

        # Compare belief
        # Reshape (B,T,C,H,W) --> (B*T,C,H,W) to compare individual images
        # Reverse the sign to maximize SSIM loss
        loss_belief = -self.belief_criterion(
            hallu_current.reshape([-1] + self.attention_dim),
            attn_future.reshape([-1] + self.attention_dim),
        )
        return loss_belief

    def freeze_fn(self, freeze_mode):
        self.feat_model.freeze_fn(freeze_mode)

    @property
    def input_mean(self):
        return self.feat_model.input_mean

    @property
    def input_std(self):
        return self.feat_model.input_std

    @property
    def crop_size(self):
        return self.feat_model.input_size

    @property
    def scale_size(self):
        return self.feat_model.scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
