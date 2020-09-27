"""Action recognition using GRU
"""
from torch import nn
from torch.nn.init import normal_, constant_

from .base_model import BaseModel


class ActregGRU2(BaseModel):

    def __init__(self, device, modality, num_class, dropout, feature_dim,
                 rnn_input_size, rnn_hidden_size, rnn_num_layers):
        super(ActregGRU2, self).__init__(device)

        self.modality = modality
        self.num_class = num_class
        self.dropout = dropout

        self.feature_dim = feature_dim
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        # Prepare some generic layers and variables
        self.relu = nn.ReLU()
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        _std = 0.001

        # Fusion layer for multiple modalities
        self.fc1 = nn.Linear(len(modality)*feature_dim, rnn_input_size)
        normal_(self.fc1.weight, 0, _std)
        constant_(self.fc1.bias, 0)

        # RNN
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        ).to(self.device)

        # Classification layers
        if isinstance(num_class, (list, tuple)):
            self.fc_verb = nn.Linear(rnn_hidden_size, num_class[0])
            self.fc_noun = nn.Linear(rnn_hidden_size, num_class[1])
            normal_(self.fc_verb.weight, 0, _std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, _std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(rnn_hidden_size, num_class)
            normal_(self.fc_action.weight, 0, _std)
            constant_(self.fc_action.bias, 0)

    def forward(self, x, hidden=None):
        """
        Args:
            x: input tensor of shape (B, T, D)
            hidden: hidden memory
        """
        self.rnn.flatten_parameters()

        # Fusion with fc layer
        x = self.relu(self.fc1(x))

        # RNN
        x, hidden = self.rnn(x, hidden)  # Let GRU handles if hidden is None (zero-init)
        x = self.relu(x)

        # Classification
        if self.dropout > 0:
            x = self.dropout_layer(x)

        if isinstance(self.num_class, (list, tuple)):
            out_verb = self.fc_verb(x)
            out_noun = self.fc_noun(x)
            output = (out_verb, out_noun)
        else:
            output = self.fc_action(x)
        return output, hidden
