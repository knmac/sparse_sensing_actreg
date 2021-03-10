"""Action recognition using GRU with Multihead, each head is ActregGRU2
"""
import os

import torch

from .base_model import BaseModel
from .actreg_gru2 import ActregGRU2


class ActregGRU3(BaseModel):

    def __init__(self, device, modality, num_class, dropout,
                 dim_global, dim_local, rnn_hidden_size, rnn_num_layers,
                 consensus_type, weight_global, weight_local, weight_both,
                 pretrained_global, pretrained_local, pretrained_both):
        super(ActregGRU3, self).__init__(device)

        self.modality = modality
        self.num_class = num_class
        self.dropout = dropout

        self.dim_global = dim_global
        self.dim_local = dim_local
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.consensus_type = consensus_type

        self.weight_global = weight_global
        self.weight_local = weight_local
        self.weight_both = weight_both

        opts = {
            'device': device,
            'modality': modality,
            'num_class': num_class,
            'dropout': dropout,
            'feature_dim': 0,  # Use extra_dim for more flexibility
            'rnn_input_size': None,  # Don't use fusion fc1
            'rnn_hidden_size': rnn_hidden_size,
            'rnn_num_layers': rnn_num_layers,
            'consensus_type': consensus_type,
        }
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if weight_global > 0:
            self.actreg_global = ActregGRU2(extra_dim=dim_global, **opts)
            if pretrained_global is not None:
                if not os.path.isfile(pretrained_global):
                    pretrained_global = os.path.join(root, pretrained_global)
                self.actreg_global.load_model(pretrained_global)

        if weight_local > 0:
            self.actreg_local = ActregGRU2(extra_dim=dim_local, **opts)
            if pretrained_local is not None:
                if not os.path.isfile(pretrained_local):
                    pretrained_local = os.path.join(root, pretrained_local)
                self.actreg_local.load_model(pretrained_local)

        if weight_both > 0:
            self.actreg_both = ActregGRU2(extra_dim=dim_local+dim_global, **opts)
            if pretrained_both is not None:
                if not os.path.isfile(pretrained_both):
                    pretrained_both = os.path.join(root, pretrained_both)
                self.actreg_both.load_model(pretrained_both)

    def forward(self, x, hidden=None):
        """
        Args:
            x: concatenated input with the order (in channel dimension):
                [global_feat, local_feat]
            hidden: hidden memory. Either None or list of hidden memory for
                each branch
        """
        if hidden is None:
            hidden_global, hidden_local, hidden_both = None, None, None
        else:
            hidden_global, hidden_local, hidden_both = hidden

        # Process concatenated input
        x_global = x[..., :self.dim_global]
        x_local = x[..., self.dim_global:]
        assert x_local.shape[-1] == self.dim_local

        # Multi head RNNs
        output = []

        if self.weight_global > 0:
            output_global, hidden_global = self.actreg_global(x_global, hidden_global)
            output += [output_global, self.repeat_weight(self.weight_global, x)]

        if self.weight_local > 0:
            output_local, hidden_local = self.actreg_local(x_local, hidden_local)
            output += [output_local, self.repeat_weight(self.weight_local, x)]

        if self.weight_both > 0:
            output_both, hidden_both = self.actreg_both(x, hidden_both)
            output += [output_both, self.repeat_weight(self.weight_both, x)]

        hidden = [hidden_global, hidden_local, hidden_both]
        return tuple(output), hidden

    def repeat_weight(self, weight, x):
        batch_size = x.shape[0]
        return torch.repeat_interleave(torch.tensor([[weight]]),
                                       batch_size, dim=0).to(x.device)
