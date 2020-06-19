"""Test model"""
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory


class TestModel(unittest.TestCase):
    """Test model loading"""

    def test_tbn_model(self):
        """Test TBN model"""
        model_cfg = 'configs/model_cfgs/tbn.yaml'
        model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
        model_factory = ModelFactory()

        # Build TBN model
        device = torch.device('cuda')
        model = model_factory.generate(model_name, device=device, **model_params)
        model.to(device)

        # Forward a random input
        sample = {
            'RGB': torch.rand([1, 9, 224, 224]).to(device),
            'Flow': torch.rand([1, 30, 224, 224]).to(device),
            'Spec': torch.rand([1, 3, 256, 256]).to(device),
        }
        model(sample)


if __name__ == '__main__':
    unittest.main()
