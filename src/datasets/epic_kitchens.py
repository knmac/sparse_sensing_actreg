"""Epic kitchen dataset"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.base_dataset import BaseDataset


class EpicKitchenDataset(BaseDataset):
    def __init__(self, mode):
        super().__init__(mode)

    def __len__(self):
        # TODO: implement this
        raise NotImplementedError
        return 0

    def __getitem__(self, idx):
        # TODO: implement this
        raise NotImplementedError
        return
