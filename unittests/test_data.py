"""Test datasets"""
import sys
import os
import unittest

import torchvision

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import DatasetFactory
from src.utils.transforms import (
    GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupScale,
    GroupCenterCrop, GroupNormalize, IdentityTransform,
    Stack, ToTorchFormatTensor
)


def get_train_augmentation(modality, input_size):
    """Copied from epic fusion model's train augmentation"""
    augmentation = {}
    if 'RGB' in modality:
        augmentation['RGB'] = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(input_size['RGB'], [1, .875, .75, .66]),
             GroupRandomHorizontalFlip(is_flow=False)])
    if 'Flow' in modality:
        augmentation['Flow'] = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(input_size['Flow'], [1, .875, .75]),
             GroupRandomHorizontalFlip(is_flow=True)])
    if 'RGBDiff' in modality:
        augmentation['RGBDiff'] = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(input_size['RGBDiff'], [1, .875, .75]),
             GroupRandomHorizontalFlip(is_flow=False)])
    return augmentation


def get_transforms(modality, input_mean, input_std, scale_size, crop_size,
                   train_augmentation, flow_prefix='', arch='BNInception'):
    """Copied from epic fusion train.py"""
    normalize = {}
    for m in modality:
        if (m != 'Spec'):
            if (m != 'RGBDiff'):
                normalize[m] = GroupNormalize(input_mean[m], input_std[m])
            else:
                normalize[m] = IdentityTransform()

    image_tmpl = {}
    train_transform = {}
    val_transform = {}
    for m in modality:
        if (m != 'Spec'):
            # Prepare dictionaries containing image name templates for each modality
            if m in ['RGB', 'RGBDiff']:
                image_tmpl[m] = "img_{:010d}.jpg"
            elif m == 'Flow':
                image_tmpl[m] = flow_prefix + "{}_{:010d}.jpg"
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = torchvision.transforms.Compose([
                train_augmentation[m],
                Stack(roll=(arch == 'BNInception')),
                ToTorchFormatTensor(div=(arch != 'BNInception')),
                normalize[m],
            ])

            val_transform[m] = torchvision.transforms.Compose([
                GroupScale(int(scale_size[m])),
                GroupCenterCrop(crop_size[m]),
                Stack(roll=(arch == 'BNInception')),
                ToTorchFormatTensor(div=(arch != 'BNInception')),
                normalize[m],
            ])
        else:
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = torchvision.transforms.Compose([
                Stack(roll=(arch == 'BNInception')),
                ToTorchFormatTensor(div=False),
            ])

            val_transform[m] = torchvision.transforms.Compose([
                Stack(roll=(arch == 'BNInception')),
                ToTorchFormatTensor(div=False),
            ])
    return train_transform, val_transform


class TestData(unittest.TestCase):
    """Test dataset loading"""

    def test_epic_kitchens(self):
        """Test epic kitchens dataset"""
        dataset_cfg = 'configs/dataset_cfgs/epickitchens_action.yaml'
        dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
        dataset_factory = DatasetFactory()

        # Prepare some extra parameters
        # TODO: modularize this
        modality = dataset_params['modality']
        crop_size = {'RGB': 224, 'Flow': 224, 'Spec': 224}
        train_transform, val_transform = get_transforms(
            modality=modality,
            input_mean={'RGB': [104, 117, 128], 'Flow': [128]},
            input_std={'RGB': [1], 'Flow': [1], 'Spec': [1]},
            scale_size={'RGB': 256, 'Flow': 256, 'Spec': 256},
            crop_size=crop_size,
            train_augmentation=get_train_augmentation(modality, crop_size),
        )

        new_length = {'RGB': 1, 'Flow': 5, 'Spec': 1}

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


if __name__ == '__main__':
    unittest.main()
