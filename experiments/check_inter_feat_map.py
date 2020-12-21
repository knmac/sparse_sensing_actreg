"""Check the intermediate feature maps of SAN
"""
import os
import sys

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

sys.path.insert(0, os.path.abspath('.'))

from src.factories import ModelFactory, DatasetFactory
from src.utils.load_cfg import ConfigLoader
from src.utils.misc import MiscUtils


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_CFG = './configs/dataset_cfgs/epickitchens_short.yaml'
MODEL_CFG = './configs/model_cfgs/pipeline_simple_san19pair_rgbspec_epicpretrained.yaml'
TRAIN_CFG = './configs/train_cfgs/train_tbn_short.yaml'
N_FRAMES = 10


def load(model_cfg, dataset_cfg, train_cfg, new_num_segments=None):
    # Load configurations
    model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
    train_params = ConfigLoader.load_train_cfg(train_cfg)

    if new_num_segments is not None:
        model_params['num_segments'] = new_num_segments

    dataset_params.update({
        'modality': model_params['modality'],
        'num_segments': model_params['num_segments'],
        'new_length': model_params['new_length'],
    })

    # Build model
    model_factory = ModelFactory()
    model = model_factory.generate(model_name, device=DEVICE,
                                   model_factory=model_factory, **model_params)
    # model.load_model(weight)
    model = model.to(DEVICE)

    # Get training augmentation and transforms
    train_augmentation = MiscUtils.get_train_augmentation(model.modality, model.crop_size)
    train_transform, val_transform = MiscUtils.get_train_val_transforms(
        modality=model.modality,
        input_mean=model.input_mean,
        input_std=model.input_std,
        scale_size=model.scale_size,
        crop_size=model.crop_size,
        train_augmentation=train_augmentation,
    )

    # Data loader
    dataset_factory = DatasetFactory()
    loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': train_params['num_workers'],
        'pin_memory': True,
    }

    val_dataset = dataset_factory.generate(dataset_name, mode='val',
                                           transform=val_transform, **dataset_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    return model, val_loader


def forward_wrap(model, x):
    """Wrapper of the forward function in SAN to retrieve intermediate feature maps
    """
    out_feats = {}
    x = model.relu(model.bn_in(model.conv_in(x)))
    out_feats['in'] = x
    x = model.relu(model.bn0(model.layer0(model.conv0(model.pool(x)))))
    out_feats['0'] = x
    x = model.relu(model.bn1(model.layer1(model.conv1(model.pool(x)))))
    out_feats['1'] = x
    x = model.relu(model.bn2(model.layer2(model.conv2(model.pool(x)))))
    out_feats['2'] = x
    x = model.relu(model.bn3(model.layer3(model.conv3(model.pool(x)))))
    out_feats['3'] = x
    x = model.relu(model.bn4(model.layer4(model.conv4(model.pool(x)))))
    out_feats['4'] = x

    # Use the avgpool layer, if not removed yet
    if hasattr(model, 'avgpool'):
        x = model.avgpool(x)
        x = x.view(x.size(0), -1)

    # Use the final fc layer, if not removed yet
    if hasattr(model, 'fc'):
        x = model.fc(x)
    return x, out_feats


def main():
    # Get model and data loader
    model, val_loader = load(
        MODEL_CFG, DATASET_CFG, TRAIN_CFG, new_num_segments=N_FRAMES)
    model.eval()
    rgb_model = model.light_model.rgb

    # Get a sample
    sample_idx = 64
    sample, _ = val_loader.dataset[sample_idx]
    sample = sample['RGB'].view(N_FRAMES, 3, 224, 224).to(DEVICE)

    # Forward wrapper
    with torch.no_grad():
        _, out_feats = forward_wrap(rgb_model, sample)

    # Visualize
    key_lst = ['in', '0', '1', '2', '3', '4']
    fig, axes = plt.subplots(len(key_lst), N_FRAMES, figsize=(20, 20))
    for i, key in enumerate(key_lst):
        feat = out_feats[key].mean(dim=1).cpu().numpy()
        vmin, vmax = feat.min(), feat.max()
        for j in range(N_FRAMES):
            axes[i, j].imshow(feat[j], vmin=vmin, vmax=vmax)
            axes[-1, j].set_xlabel('t={}'.format(j))
        axes[i, 0].set_ylabel('layer={}'.format(key))
    plt.show()


if __name__ == '__main__':
    main()
