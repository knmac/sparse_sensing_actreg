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
    def __init__(self, **kwargs):
        kwargs.update({'modality': ['RGBDS']})
        super(EpicWrapper, self).__init__(**kwargs)

    def __getitem__(self, index):
        record = self.video_list[index]
        indices = np.arange(record.num_frames['RGBDS'])
        self.get('RGBDS', record, indices)

    def get(self, modality, record, indices):
        """Get sample based on the given modality
        """
        images = list()

        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_data(modality, record, p)

            # Skip if the data are not loaded
            if seg_imgs is None:
                continue

            # Collect results
            images.extend(seg_imgs)
            if p < record.num_frames[modality]:
                p += 1

        # Transform data
        # process_data = self.transform[modality](images)
        # return process_data, record.label
        # return images, record.label


def process(dataset_cfg, n_threads, mode):
    print('='*30 + '\n' + mode + '\n' + '='*30)
    _, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)

    # Create dataset
    dataset_params.update({
        'mode': mode,
        'modality': ['RGBDS'],
        'new_length': {'RGBDS': 1},
    })
    dataset = EpicWrapper(**dataset_params)

    loader_params = {
        'batch_size': n_threads,
        'num_workers': n_threads*2,
        'pin_memory': True,
        'collate_fn': MiscUtils.safe_collate,  # safely remove broken samples
    }
    loader = DataLoader(dataset, shuffle=False, **loader_params)

    for i, data in enumerate(loader):
        print('{}/{}'.format(i, len(loader)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_threads', type=int, default=48)
    parser.add_argument('-d', '--dataset_cfg', type=str,
                        default='configs/dataset_cfgs/epickitchens_noshuffle_rgbds.yaml')
                        # default='configs/dataset_cfgs/epickitchens_p01_08.yaml')
    args = parser.parse_args()

    process(args.dataset_cfg, args.n_threads, 'train')
    process(args.dataset_cfg, args.n_threads, 'val')
    process(args.dataset_cfg, args.n_threads, 'test')
    return 0


if __name__ == '__main__':
    sys.exit(main())
