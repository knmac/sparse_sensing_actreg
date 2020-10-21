"""Misc helper methods"""
import os
import sys
import glob
import shutil
import argparse

import torch
import torchvision

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.transforms import (
    GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupScale,
    GroupCenterCrop, GroupNormalize, IdentityTransform,
    Stack, ToTorchFormatTensor
)
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class MiscUtils:

    @staticmethod
    def save_progress(model, optimizer, logdir, best_val, epoch, is_best=False):
        """Save the training progress for model and optimizer

        Data are saved as: [logdir]/epoch_[epoch].[extension]

        Args:
            model: model to save
            optimizer: optimizer to save
            logdir: where to save data
            epoch: the current epoch
            is_best: if True, will backup the best model
        """
        prefix = os.path.join(logdir, 'epoch_{:05d}'.format(epoch))
        logger.info('Saving to: %s' % prefix)

        try:
            model.save_model(prefix+'.model')
        except AttributeError:
            model.module.save_model(prefix+'.model')
        # torch.save(optimizer.state_dict(), prefix+'.opt')
        # torch.save(torch.get_rng_state(), prefix+'.rng')
        # torch.save(torch.cuda.get_rng_state(), prefix+'.curng')

        data = {
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'best_prec1': best_val,
        }
        torch.save(data, prefix+'.stat')

        if is_best:
            shutil.copyfile(os.path.join(prefix+'.model'),
                            os.path.join(logdir, 'best.model'))

    @staticmethod
    def load_progress(model, optimizer, device, prefix):
        """Load the training progress for model and optimizer

        Data are loaded from: [prefix].[extension]

        Args:
            model: model to load
            optimizer: optimizer to load
            deivce: device to transfer the optimizer states to
            prefix: prefix with the format [logdir]/epoch_[epoch]

        Return:
            lr: loaded learning rate
            next_epoch: id of the nex epoch
        """
        logger.info('Loading from: %s' % prefix)

        model.load_model(prefix+'.model')
        # optimizer.load_state_dict(torch.load(prefix+'.opt'))
        # torch.set_rng_state(torch.load(prefix+'.rng'))
        # torch.cuda.set_rng_state(torch.load(prefix+'.curng'))

        data = torch.load(prefix+'.stat')
        optimizer.load_state_dict(data['optimizer'])
        torch.set_rng_state(data['rng_state'])
        torch.cuda.set_rng_state(data['cuda_rng_state'])
        best_val = data['best_prec1']

        lr = optimizer.param_groups[0]['lr']
        tmp = os.path.basename(prefix)
        next_epoch = int(tmp.replace('epoch_', '')) + 1

        # Individually transfer the optimizer parts
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return lr, next_epoch, best_val

    @staticmethod
    def get_lastest_checkpoint(logdir, regex='epoch_*.model'):
        """Get the latest checkpoint in a logdir

        For example, for the logdir with:
            logdir/
                epoch_00000.model
                epoch_00001.model
                ...
                epoch_00099.model
        The function will return `logdir/epoch_00099`

        Args:
            logdir: log directory to find the latest checkpoint
            regex: regular expression to describe the checkpoint

        Return:
            prefix: prefix with the format [logdir]/epoch_[epoch]
        """
        assert os.path.isdir(logdir), 'Not a directory: {}'.format(logdir)

        save_lst = glob.glob(os.path.join(logdir, regex))
        save_lst.sort()
        prefix = save_lst[-1].replace('.model', '')
        return prefix

    @staticmethod
    def str2bool(v):
        """Convert a string to boolean type for argparse"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def get_train_augmentation(modality, input_size):
        """Get data augmentation for training phase.
        Copied from epic fusion model's train augmentation.

        Args:
            modality: (dict) dictionary of modality to add transformation for
                augmentation. Different modality requires different transform
            input_size: (dict) dictionary of input size for each modality. We
                assume that the width and height are the same. For example,
                {'RGB': 224, 'Flow': 224, 'Spec': 224}

        Return:
            augmentation: (dict) dictionary of composed transforms for each
                modality
        """
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

    @staticmethod
    def get_train_val_transforms(modality, input_mean, input_std, scale_size,
                                 crop_size, train_augmentation,
                                 flow_prefix='', arch='BNInception'):
        """Get transform for train and val phases.
        Copied from epic fusion train.py

        Args:
            modality: (dict) dictionary of modality to add transformation for
                augmentation. Different modality requires different transform
            input_mean: (dict) mean per channels for different modalities, only
                for 'RGB' and 'Flow' modalities, e.g.
                {RGB': [104, 117, 128], 'Flow': [128]}
            input_std: (dict) standard deviation for different modalities, e.g.,
                {'RGB': [1], 'Flow': [1], 'Spec': [1]}
            scale_size: (dict) size to rescale the input to before cropping, e.g.
                {'RGB': 256, 'Flow': 256, 'Spec': 256}
            train_augmentation: (dict) extra augmentation for training phase.
                Refer to get_train_augmentation() here for more information
            flow_prefix: (str) prefix of flow filenames for loading
            arch: (str) name of the architecture. Add special transformations if
                arch == BNInception

        Return:
            train_transform, val_transform: (dict, dict) transformations wrt
                different modalities for training and validation
        """
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
