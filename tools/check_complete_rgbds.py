import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm

from src.datasets.epic_kitchens import EpicKitchenDataset
from src.utils.load_cfg import ConfigLoader


class EpicWrapper(EpicKitchenDataset):
    def __init__(self, n_parts, part_id, **kwargs):
        kwargs.update({'modality': ['RGBDS']})

        self.n_parts = n_parts
        self.part_id = part_id
        super(EpicWrapper, self).__init__(**kwargs)

    def __getitem__(self, index):
        record = self.video_list[index]
        indices = np.arange(record.num_frames['RGBDS'])
        n_total, n_complete = 0, 0
        missing = set()
        for idx in indices:
            res, pth = self._load_rgbds(record, idx)  # Force RGBDS
            if res is not None:
                n_total += 1
                if res is True:
                    n_complete += 1
                else:
                    missing.add(pth)

        return n_total, n_complete, record.untrimmed_video_name, missing

    def _load_rgbds(self, record, idx):
        # Prepare the file paths and load neighbors if not found---------------
        idx_untrimmed = record.start_frame + idx
        found = False
        _offset = 0
        while idx_untrimmed <= record.end_frame:
            idx_untrimmed = record.start_frame + idx + _offset
            inliers_pth = os.path.join(
                self.depth_path, record.untrimmed_video_name,
                self.depth_tmpl.format(idx_untrimmed-1))
            if os.path.isfile(inliers_pth):
                found = True
                break
            _offset += 1

        _offset = 0
        while idx_untrimmed >= record.start_frame:
            if found:
                break
            idx_untrimmed = record.start_frame + idx - _offset
            inliers_pth = os.path.join(
                self.depth_path, record.untrimmed_video_name,
                self.depth_tmpl.format(idx_untrimmed-1))
            if os.path.isfile(inliers_pth):
                found = True
                break
            _offset += 1

        # Missing key frames for the whole video sequence -> skip
        if not found:
            return None, ''

        # Try to load cached depth and semantic images, if available ----------
        depth_cache_pth = os.path.join(
            self.depth_cache_tmpl.format(record.untrimmed_video_name, idx_untrimmed-1))
        semantic_cache_pth = os.path.join(
            self.semantic_cache_tmpl.format(record.untrimmed_video_name, idx_untrimmed-1))
        return os.path.isfile(depth_cache_pth) and os.path.isfile(semantic_cache_pth), depth_cache_pth

    def _parse_list(self):
        """Parse from pandas data frame to list of EpicVideoRecord objects"""
        super(EpicWrapper, self)._parse_list()

        # Cut down the list
        self.video_list = [self.video_list[i] for i in range(len(self.video_list))
                           if i % self.n_parts == self.part_id]


def process(dataset_cfg, n_parts, part_id, output, mode):
    print('='*30 + '\n' + mode + '\n' + '='*30)
    _, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)

    # Create dataset
    dataset_params.update({
        'mode': mode,
        'modality': ['RGBDS'],
        'new_length': {'RGBDS': 1},
    })
    dataset = EpicWrapper(n_parts, part_id, **dataset_params)

    all_total, all_complete = 0, 0
    all_missing = set()
    stats = {}
    for i in tqdm(range(len(dataset))):
        n_total, n_complete, vid, missing = dataset[i]
        if vid not in stats:
            stats[vid] = {'n_total': 0, 'n_complete': 0}
        stats[vid]['n_total'] += n_total
        stats[vid]['n_complete'] += n_complete
        all_total += n_total
        all_complete += n_complete
        all_missing |= missing

    for vid in stats:
        print(f"{vid}: {100*stats[vid]['n_complete']/stats[vid]['n_total']:.02f}%")
    print(f"Overall: {100*all_complete/all_total:.02f}%")

    with open(f"{output}.{mode}", 'w') as f:
        for item in all_missing:
            f.write(item+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_cfg', type=str,
                        default='configs/dataset_cfgs/epickitchens_noshuffle_rgbds.yaml',
                        help='Dataset configuration')
    parser.add_argument('-o', '--output', type=str,
                        default='missing',
                        help='Output the missing files')
    args = parser.parse_args()

    process(args.dataset_cfg, 1, 0, args.output, 'train')
    process(args.dataset_cfg, 1, 0, args.output, 'val')
    process(args.dataset_cfg, 1, 0, args.output, 'test')
    return 0a


if __name__ == '__main__':
    sys.exit(main())
