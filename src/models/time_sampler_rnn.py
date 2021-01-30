"""Temporal sampler with trainable RNN policy
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseModel
from .pytorch_ssim.ssim import SSIM


class TemporalSamplerRNN(BaseModel):
    def __init__(self, device, attention_dim, max_frames_skip,
                 rnn_input_size, rnn_hidden_size, rnn_num_layers,
                 use_attn, use_hallu, use_ssim):
        super(TemporalSamplerRNN, self).__init__(device)

        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.attention_dim = attention_dim
        self.max_frames_skip = max_frames_skip  # Maximum number of frames that can be skipped
        self.use_attn = use_attn
        self.use_hallu = use_hallu
        self.use_ssim = use_ssim

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        ).to(self.device)

        assert use_attn or use_hallu or use_ssim, 'Must use at least 1 input'
        input_dim = 0
        if use_attn:
            input_dim += np.prod(attention_dim)
        if use_hallu:
            input_dim += np.prod(attention_dim)
        if use_ssim:
            input_dim += 1
            self.belief_criterion = SSIM(window_size=3, channel=attention_dim[0])

        self.fc_fus = nn.Linear(input_dim, rnn_input_size)
        self.fc_out = nn.Linear(rnn_hidden_size, max_frames_skip+1)  # Include no skipping
        self.softmax = nn.Softmax(dim=1)

        self.old_hidden = None  # old hidden memory

    def reset(self):
        """Reset memory
        """
        self.old_hidden = None

    def forward(self, dummy):
        """Dummy function to compute complexity"""
        self.sample_frame(dummy[:, 0], dummy[:, 1], 5.)

    def sample_frame(self, attn, old_hallu, temperature):
        """Decide how many frames to skip
        """
        self.rnn.flatten_parameters()

        # Prepare input
        x = []
        if self.use_attn:
            x.append(attn.flatten(start_dim=1))
        if self.use_hallu:
            if old_hallu is not None:
                x.append(old_hallu.flatten(start_dim=1))
            else:
                x.append(torch.zeros([1, np.prod(self.attention_dim)]).to(self.device))
        if self.use_ssim:
            if old_hallu is not None:
                ssim = -self.belief_criterion(attn, old_hallu).unsqueeze(0).unsqueeze(0)
            else:
                ssim = torch.Tensor([[0.0]]).to(self.device)
            x.append(ssim)
        x = torch.cat(x, dim=1).unsqueeze(dim=1)  # (N, 1, C) -> sequence of 1

        # Feed to RNN
        x = self.fc_fus(x)
        x, hidden = self.rnn(x, self.old_hidden)
        self.old_hidden = hidden
        out = self.fc_out(x).squeeze(dim=1)

        # Sampling from RNN output
        p_t = torch.log(self.softmax(out).clamp(min=1e-8))
        r_t = torch.cat([F.gumbel_softmax(p_t[b_i:b_i + 1], tau=temperature, hard=True) for b_i in range(p_t.shape[0])])

        return r_t, ssim
