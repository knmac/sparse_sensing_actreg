"""Test datasets"""
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils


class TestData(unittest.TestCase):
    """Test dataset loading"""

    def test_epic_kitchens(self):
        """Test epic kitchens dataset"""
        dataset_cfg = 'configs/dataset_cfgs/epickitchens_action.yaml'
        dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
        dataset_factory = DatasetFactory()

        # Prepare some extra parameters
        modality = dataset_params['modality']
        input_mean = {'RGB': [104, 117, 128], 'Flow': [128]}
        input_std = {'RGB': [1], 'Flow': [1], 'Spec': [1]}
        scale_size = {'RGB': 256, 'Flow': 256, 'Spec': 256}
        crop_size = {'RGB': 224, 'Flow': 224, 'Spec': 224}
        new_length = {'RGB': 1, 'Flow': 5, 'Spec': 1}

        # Get augmentation and transforms
        train_augmentation = MiscUtils.get_train_augmentation(modality, crop_size)
        train_transform, val_transform = MiscUtils.get_train_val_transforms(
            modality=modality,
            input_mean=input_mean,
            input_std=input_std,
            scale_size=scale_size,
            crop_size=crop_size,
            train_augmentation=train_augmentation,
        )

        # Create dataset
        dataset = dataset_factory.generate(
            dataset_name, mode='val', new_length=new_length,
            transform=val_transform, **dataset_params,
        )

        sample, label = dataset[0]
        # Check shape
        assert label['verb'] == 2
        assert label['noun'] == 10
        assert sample['RGB'].numpy().shape == (9, 224, 224)
        assert sample['Flow'].numpy().shape == (30, 224, 224)
        assert sample['Spec'].numpy().shape == (3, 256, 256)

        # Check sum
        assert (sample['RGB'].numpy().sum() - -15417859.0) < 0.1
        assert (sample['Flow'].numpy().sum() - 12470523.0) < 0.1
        assert (sample['Spec'].numpy().sum() - -1589548.8) < 0.1

        # Check with data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False,
            num_workers=4, pin_memory=True)

        print('')
        for i, (samples, labels) in enumerate(data_loader):
            print(i, samples['RGB'].shape, samples['Flow'].shape, samples['Spec'].shape)
            if i >= 2:
                break


if __name__ == '__main__':
    unittest.main()
