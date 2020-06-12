"""Misc helper methods"""
import os
import sys
import glob
import argparse

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class MiscUtils:

    @staticmethod
    def save_progress(model, optimizer, logdir, epoch):
        """Save the training progress for model and optimizer

        Data are saved as: [logdir]/epoch_[epoch].[extension]

        Args:
            model: model to save
            optimizer: optimizer to save
            logdir: where to save data
            epoch: the current epoch
        """
        prefix = os.path.join(logdir, 'epoch_{:05d}'.format(epoch))
        logger.info('Saving to: %s' % prefix)

        model.save_model(prefix+'.model')
        torch.save(optimizer.state_dict(), prefix+'.opt')
        torch.save(torch.get_rng_state(), prefix+'.rng')
        torch.save(torch.cuda.get_rng_state(), prefix+'.curng')

    @staticmethod
    def load_progress(model, optimizer, prefix):
        """Load the training progress for model and optimizer

        Data are loaded from: [prefix].[extension]

        Args:
            model: model to load
            optimizer: optimizer to load
            prefix: prefix with the format [logdir]/epoch_[epoch]

        Return:
            lr: loaded learning rate
            next_epoch: id of the nex epoch
        """
        logger.info('Loading from: %s' % prefix)

        model.load_model(prefix+'.model')
        optimizer.load_state_dict(torch.load(prefix+'.opt'))
        torch.set_rng_state(torch.load(prefix+'.rng'))
        torch.cuda.set_rng_state(torch.load(prefix+'.curng'))

        lr = optimizer.param_groups[0]['lr']
        tmp = os.path.basename(prefix)
        next_epoch = int(tmp.replace('epoch_', '')) + 1
        return lr, next_epoch

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
