import sys
import os

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.utils.load_cfg import ConfigLoader
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils


def main():
    """Test epic kitchens dataset with only rgbds and spectrogram"""
    dataset_cfg = 'configs/dataset_cfgs/epickitchens_short.yaml'
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
    dataset_factory = DatasetFactory()
    dataset_params.update({'has_motion_compensation': True})

    # Prepare some extra parameters
    modality = ['RGB']
    num_segments = 10
    input_mean = {'RGB': [104, 117, 128]}
    input_std = {'RGB': [1]}
    scale_size = {'RGB': 256}
    crop_size = {'RGB': 224}
    new_length = {'RGB': 1}

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
    dataset_params['list_file']['val'] = 'dataset_splits/EPIC_KITCHENS_2018/P01_08.pkl'
    dataset = dataset_factory.generate(
        dataset_name, mode='val', modality=modality,
        num_segments=num_segments, new_length=new_length,
        transform=val_transform, **dataset_params,
    )

    # Get data
    idx = 0
    sample, _ = dataset.__getitem__(idx)
    dataset.has_motion_compensation = False
    sample_orig, _ = dataset.__getitem__(idx)

    # Visualize
    rgb = MiscUtils.deprocess_rgb(sample['RGB'], num_segments)
    rgb_orig = MiscUtils.deprocess_rgb(sample_orig['RGB'], num_segments)

    fig, axes = plt.subplots(2, num_segments, figsize=(15, 3))
    for i in range(num_segments):
        axes[0, i].imshow(rgb_orig[i])
        axes[1, i].imshow(rgb[i])

    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Warped')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


if __name__ == '__main__':
    main()
