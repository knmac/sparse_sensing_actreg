"""Pipeline version 4 - Run only action recognition
"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from .base_model import BaseModel
from src.utils.load_cfg import ConfigLoader


class Pipeline4(BaseModel):
    def __init__(self, device, model_factory, num_class, modality, num_segments,
                 new_length, dropout, feat_model_cfg, actreg_model_cfg):
        super(Pipeline4, self).__init__(device)

        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.dropout = dropout

        # Generate feature extraction model
        name, params = ConfigLoader.load_model_cfg(feat_model_cfg)
        params.update({
            'new_length': self.new_length,
            'modality': self.modality,
        })
        self.feat_model = model_factory.generate(name, device=device, **params)

        # Generate action recognition model
        name, params = ConfigLoader.load_model_cfg(actreg_model_cfg)
        assert name in ['ActregGRU2'], \
            'Unsupported model: {}'.format(name)
        params.update({
            'feature_dim': self.feat_model.feature_dim,
            'modality': self.modality,
            'num_class': self.num_class,
            'dropout': self.dropout,
        })
        self.actreg_model = model_factory.generate(name, device=device, **params)

    def forward(self, x):
        # Extract features
        batch_size = x['RGB'].shape[0]
        x = self.feat_model(x)

        # (B*T, C) --> (B, T, C)
        x = x.view([batch_size,
                    self.num_segments,
                    len(self.modality)*self.feat_model.feature_dim])

        # Action recognition
        hidden = None
        output, hidden = self.actreg_model(x, hidden)

        # Does not need belief_loss because the function compare_belief is not
        # available here
        return output

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
