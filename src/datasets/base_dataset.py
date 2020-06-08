"""Base dataset"""
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset to inherit from"""
    def __init__(self, mode):
        assert mode in ['train', 'val', 'test'], \
                'Unsupported mode: {}'.format(mode)
