"""Main code for train/val/test"""
import sys
import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from train_val import train_val, validate
from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()

    # Config files
    parser.add_argument('-c', '--model_cfg', type=str,
                        help='Path to the model config filename')
    parser.add_argument('-d', '--dataset_cfg', type=str,
                        help='Path to the dataset config filename')
    parser.add_argument('-t', '--train_cfg', type=str,
                        help='Path to the training config filename')

    # Runtime configs
    parser.add_argument('-i', '--is_training', type=MiscUtils.str2bool,
                        help='Whether is in training or testing mode')
    parser.add_argument('-m', '--train_mode', type=str,
                        choices=['from_scratch', 'from_pretrained', 'resume'],
                        help='Which mode to start the training from')
    parser.add_argument('-l', '--logdir', type=str, default='logs',
                        help='Directory to store the log')
    parser.add_argument('--log_fname', type=str, default=None,
                        help='Path to the file to store running log '
                             '(beside showing to stdout)')

    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--experiment_suffix', type=str, default='',
                        help='Addtional suffix for experiment')

    # parser.add_argument(
    #     '--pretrained_model_path', type=str, default='',
    #     help='Path to the model to test. Only needed if is not training or '
    #          'is training and mode==from_pretrained')

    args = parser.parse_args()

    # if (not args.is_training) or \
    #         (args.is_training and args.train_mode == 'from_pretrained'):
    #     assert os.path.isfile(args.pretrained_model_path), \
    #         'pretrained_model_path not found: {}'.format(args.pretrained_model_path)
    return args


def fix_seeds(seed=0):
    """Fix random seeds here for pytorch, numpy, and python"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False


def gen_experiment_name(dataset_name, model_name, model_params, train_params, args):
    """Generate experiment name and directory based on input parameters

    Args:
        dataset_name: (str) name of the dataset
        model_name: (str) name of the model
        model_params: (dict) parameters of the model
        train_params: (dict) training parameters
        args: input arguments

    Return:
        experiment_name: (str) name of the experiment
        experiment_dir: (str) experiment_name with timestamp directory
        log_dir: (str) log_dir with experiment_dir
    """
    experiment_name = '_'.join([
        dataset_name,
        model_name,
        ''.join(model_params['modality']),
        'segs' + str(model_params['num_segments']),
        'dr' + str(model_params['dropout']),
        'ep' + str(train_params['n_epochs']),
        'lr' + str(train_params['optim_params']['lr']),
        'lr_st' + '_'.join([str(x) for x in train_params['lr_steps']]),
        args.experiment_suffix,
    ])
    experiment_dir = os.path.join(experiment_name, datetime.now().strftime('%b%d_%H-%M-%S'))
    logdir = os.path.join(args.logdir, experiment_dir)
    return experiment_name, experiment_dir, logdir


def main(args):
    """Main function"""
    # Set up logging and fix random seeds
    logging.setup_logging(args.log_fname)
    fix_seeds()

    # -------------------------------------------------------------------------
    # Load configurations
    model_name, model_params = ConfigLoader.load_model_cfg(args.model_cfg)
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(args.dataset_cfg)
    train_params = ConfigLoader.load_train_cfg(args.train_cfg)

    # Copy some parameters from model to share
    dataset_params.update({
        'modality': model_params['modality'],
        'num_segments': model_params['num_segments'],
        'new_length': model_params['new_length'],
    })

    # Generate logdir with experiment name
    _, _, logdir = gen_experiment_name(
        dataset_name, model_name, model_params, train_params, args)
    args.logdir = logdir

    # Set up device, use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    # -------------------------------------------------------------------------
    # Set up common parameters for data loaders (shared among train/val/test)
    dataset_factory = DatasetFactory()
    loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': train_params['num_workers'],
        'pin_memory': True,
    }

    # Build model
    model_factory = ModelFactory()
    model = model_factory.generate(model_name, device=device, **model_params)

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

    # Set up loss criterion
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # Main pipeline
    if args.is_training:
        # Create data loader for training
        train_dataset = dataset_factory.generate(
            dataset_name, mode='train', transform=train_transform, **dataset_params)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)

        # Create data loader for validation
        val_dataset = dataset_factory.generate(
            dataset_name, mode='val', transform=val_transform, **dataset_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)

        # Train/val routine
        train_val(model, device, criterion, train_loader, val_loader, train_params, args)
        # train(model, optimizer, criterion, train_loader, val_loader, args.logdir,
        #       args.train_mode, train_params, device, args.pretrained_model_path)
    else:
        # Create data loader for testing
        test_dataset = dataset_factory.generate(
            dataset_name, mode='test', transform=val_transform, **dataset_params)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

        # Test routine
        model.load_model(args.pretrained_model_path)
        # writer = SummaryWriter(log_dir=args.logdir)
        validate(model, device, criterion, test_loader)  # TODO
        # test(model, criterion, test_loader, device, writer, 0)
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
