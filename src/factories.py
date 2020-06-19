"""Factory pattern for different models and datasets"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.epic_kitchens import EpicKitchenDataset
from src.models.tbn import TBN

import src.utils.logging as logging

logger = logging.get_logger(__name__)


class BaseFactory():
    """Base factory for dataset and model generator"""
    def __init__(self):
        self.info_msg = 'Generating object'
        self.objfn_dict = None

    def generate(self, name, **kwargs):
        """Generate object based on the given name and variables

        Args:
            name: a string to describe the type of the object
            kwargs: keyworded variables of the object to generate

        Return:
            Generated object with corresponding type and arguments
        """
        assert name in self.objfn_dict, \
            '{} not recognized. ' \
            'Only support:\n{}'.format(name, self.objfn_dict.keys())

        logger.info('%s: %s' % (self.info_msg, name))
        logger.info('Given parameters:')
        self._print_dict(kwargs, 2)
        logger.info('-'*50)

    def _print_dict(self, dict_in, indent):
        """Recursively print out a dictionary

        Args:
            dict_in: input dictionary
            indent: number of spaces as the indentation
        """
        indentation = ' '*indent
        for key, val in dict_in.items():
            if isinstance(val, dict):
                logger.info('%s%s' % (indentation, key))
                self._print_dict(val, indent+2)
            else:
                logger.info('%s%s: %s' % (indentation, key, val))


class ModelFactory(BaseFactory):
    """Factory for model generator"""
    def __init__(self):
        self.info_msg = 'Generating model'
        self.objfn_dict = {
            'TBN': TBN,
        }

    def generate(self, model_name, **kwargs):
        """Generate model based on given name and variables"""
        super().generate(model_name, **kwargs)
        gen_model = self.objfn_dict[model_name](**kwargs)
        print(gen_model)
        return gen_model


class DatasetFactory(BaseFactory):
    """Factory for dataset generator"""
    def __init__(self):
        self.info_msg = 'Generating dataset'
        self.objfn_dict = {
            'epic_kitchens': EpicKitchenDataset,
        }

    def generate(self, dataset_name, **kwargs):
        """Generate dataset based on given name and variables"""
        super().generate(dataset_name, **kwargs)
        gen_dataset = self.objfn_dict[dataset_name](**kwargs)
        return gen_dataset
