import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from torch.utils.data import DataLoader

from src.datasets.epic_kitchens import EpicKitchenDataset
from src.utils.load_cfg import ConfigLoader
from src.utils.misc import MiscUtils


class EpicWrapper(EpicKitchenDataset):
    def __init__(self, n_parts, part_id, **kwargs):
        kwargs.update({'modality': ['RGBDS']})

        self.n_parts = n_parts
        self.part_id = part_id
        super(EpicWrapper, self).__init__(**kwargs)

    def __getitem__(self, index):
        record = self.video_list[index]
        indices = np.arange(record.num_frames['RGBDS'])
        for idx in indices:
            self._load_data('RGBDS', record, idx)
        print('vid {}/{}: DONE!'.format(index, len(self.video_list)))

    def _parse_list(self):
        """Parse from pandas data frame to list of EpicVideoRecord objects"""
        super(EpicWrapper, self)._parse_list()

        # Cut down the list
        self.video_list = [self.video_list[i] for i in range(len(self.video_list))
                           if i % self.n_parts == self.part_id]


def process(dataset_cfg, n_threads, n_parts, part_id, mode):
    """Process the missing files by part"""
    print('='*30 + '\n' + mode + '\n' + '='*30)

    # Create dataset
    print('Initializing dataset...')
    _, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
    dataset_params.update({
        'mode': mode,
        'modality': ['RGBDS'],
        'new_length': {'RGBDS': 1},
    })
    if mode == 'test':  # Force the real test set
        dataset_params['list_file']['test'] = None
        dataset_params['full_test_split'] = 'all'
    dataset = EpicWrapper(n_parts, part_id, **dataset_params)

    # Use torch DataLoader for parallel processing
    loader_params = {
        'batch_size': n_threads,
        'num_workers': n_threads,
        'pin_memory': False,
        'collate_fn': MiscUtils.safe_collate,  # safely remove broken samples
    }
    loader = DataLoader(dataset, shuffle=False, **loader_params)

    # Multithread processing
    print('Start processing...')
    try:
        for i, _ in enumerate(loader):
            print('--> batch {}/{}: DONE'.format(i, len(loader)))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_threads', type=int, default=48,
                        help='Number of workers and batch size')
    parser.add_argument('-p', '--n_parts', type=int, default=14,
                        help='Total number of parts to divide the list')
    parser.add_argument('-i', '--part_id', type=int, default=0,
                        help='Part index')
    parser.add_argument('-d', '--dataset_cfg', type=str,
                        default='configs/dataset_cfgs/epickitchens_noshuffle_rgbds.yaml',
                        # default='configs/dataset_cfgs/epickitchens_p01_08.yaml',
                        help='Dataset configuration')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test', ''],
                        default='', help='Specify which mode to run, otherwise will run all')
    args = parser.parse_args()

    assert 0 <= args.part_id < args.n_parts, \
        'part_id must be in [0, n_parts)'
    print('Part {} out of {}'.format(args.part_id+1, args.n_parts))

    if args.mode == '':
        process(args.dataset_cfg, args.n_threads, args.n_parts, args.part_id, 'train')
        process(args.dataset_cfg, args.n_threads, args.n_parts, args.part_id, 'val')
        process(args.dataset_cfg, args.n_threads, args.n_parts, args.part_id, 'test')
    else:
        process(args.dataset_cfg, args.n_threads, args.n_parts, args.part_id, args.mode)

    return 0


if __name__ == '__main__':
    sys.exit(main())
