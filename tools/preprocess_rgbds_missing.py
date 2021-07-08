import sys
import os
import argparse
import math

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
import numpy as np
import skimage.io as sio
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.epic_kitchens import EpicKitchenDataset
from src.utils.load_cfg import ConfigLoader
from src.utils.misc import MiscUtils
from src.utils.read_3d_data import (read_inliner, read_intrinsic_extrinsic,
                                    project_depth, rbf_interpolate)


class EpicWrapper(EpicKitchenDataset):
    def __init__(self, n_parts, part_id, missing_list, **kwargs):
        kwargs.update({'modality': ['RGBDS']})

        self.n_parts = n_parts
        self.part_id = part_id
        self.missing_list = missing_list

        super(EpicWrapper, self).__init__(**kwargs)

    def __getitem__(self, index):
        """Wrapper for preporcess part of _load_rgbds()
        """
        # Retreve info by parsing
        depth_cache_pth = self.missing_list[index]
        untrimmed_video_name = depth_cache_pth.split('/')[-2]
        idx_untrimmed_1 = depth_cache_pth.split('/')[-1]\
            .replace('depth_', '').replace('.png', '')
        # +1 because saved idx_untrimmed-1
        idx_untrimmed = int(idx_untrimmed_1) + 1
        semantic_cache_pth = self.semantic_cache_tmpl.format(
            untrimmed_video_name, idx_untrimmed-1)

        inliers_pth = os.path.join(self.depth_path, untrimmed_video_name,
                                   self.depth_tmpl.format(idx_untrimmed-1))
        corpus_pth = os.path.join(self.depth_path, untrimmed_video_name)
        normalize_point_pth = os.path.join(self.depth_path, untrimmed_video_name,
                                           'Points.txt')

        # Load rgb image to get dimension
        rgb = Image.open(os.path.join(self.visual_path,
                                      untrimmed_video_name,
                                      self.image_tmpl['RGB'].format(idx_untrimmed))
                         ).convert('RGB')
        rgb = np.array(rgb)

        # The 4th channel: depth ----------------------------------------------
        # Get sfm_dist and real_dist
        assert os.path.isfile(normalize_point_pth)
        with open(normalize_point_pth, 'r') as fp:
            content = fp.read().splitlines()
        sfm_dist, real_dist = content[-1].split(' ')
        sfm_dist, real_dist = float(sfm_dist), float(real_dist)
        new_h, new_w = rgb.shape[0], rgb.shape[1]

        # Read depth
        ptid, pt3d, pt2d = read_inliner(inliers_pth)

        vid_info = read_intrinsic_extrinsic(
            corpus_pth, startF=idx_untrimmed-1, stopF=idx_untrimmed-1,
        ).VideoInfo
        assert vid_info is not None, 'Video information is corrupted'
        frame_info = vid_info[idx_untrimmed-1]

        if frame_info.height is None:
            frame_info.height = 1080
        if frame_info.width is None:
            frame_info.width = 1920

        cam_center = frame_info.camCenter
        principle_ray_dir = frame_info.principleRayDir
        if cam_center is None or principle_ray_dir is None:
            return None

        # Find depth wrt to camera coordinates
        depth, projection = project_depth(
            ptid, pt3d, pt2d, cam_center, principle_ray_dir,
            height=frame_info.height, width=frame_info.width,
            new_h=new_h, new_w=new_w)

        # Normalize depth to the scale in milimeters
        depth = depth / sfm_dist * real_dist

        # Interpolation
        rbf_opts = {'function': 'linear', 'epsilon': 2.0}
        depth = rbf_interpolate(depth, rbf_opts=rbf_opts)

        # Bilateral filtering with reference image
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        k_size = 9
        h_size = k_size // 2
        gray_pad = np.pad(gray, ((h_size, h_size), (h_size, h_size)))
        depth_pad = np.pad(depth, ((h_size, h_size), (h_size, h_size)))
        depth = MiscUtils.ref_bilateral_filter(depth_pad, gray_pad, 9, 3.0, 3.0)
        depth = depth[h_size:-h_size, h_size:-h_size]

        # Clip depth to [0.5m..5m] and inverse
        depth = 1000.0 / np.clip(depth, 500, 5000)

        # Normalize to 0..255
        depth = (depth - 0.2) / 1.8 * 255

        # The 5th channel: semantic -------------------------------------------
        semantic_pth = os.path.join(self.semantic_path,
                                    self.semantic_tmpl.format(untrimmed_video_name))
        assert os.path.isfile(semantic_pth), '{} not found'.format(semantic_pth)
        semantic = np.zeros(rgb.shape[:2], dtype=np.uint8)
        exclude_lst = [0, 1, 2]

        # Sort ptid by depth
        zz = [pt3d[k][2] for k in ptid]
        if zz != [] and ptid != []:
            _, ptid_sort = zip(*sorted(zip(zz, ptid), reverse=True))
        else:
            ptid_sort = []

        # Expand wrt depth
        semantic_dict = torch.load(semantic_pth)
        for k in ptid_sort:
            # Skip points that not available in semantic_dict
            if (k not in semantic_dict) or (k not in projection):
                continue

            # exclude background, hand, and floor
            if semantic_dict[k] in exclude_lst:
                continue

            u, v = projection[k]
            scale = int((pt3d[k][2] / sfm_dist * real_dist) / 30)
            semantic[max(v-scale//2, 0):min(v+scale//2, semantic.shape[0]),
                     max(u-scale//2, 0):min(u+scale//2, semantic.shape[1])
                     ] = semantic_dict[k]

        # Normalize to 0..255 (there are 0..23 classes in total)
        semantic = (semantic / 23 * 255).astype(np.uint8)

        # Save cache for depth and semantic -----------------------------------
        try:
            os.makedirs(os.path.dirname(depth_cache_pth), exist_ok=True)
            sio.imsave(depth_cache_pth, depth.astype(np.uint8), check_contrast=False)

            os.makedirs(os.path.dirname(semantic_cache_pth), exist_ok=True)
            sio.imsave(semantic_cache_pth, semantic.astype(np.uint8), check_contrast=False)
        except Exception:
            pass

        # NOTE: does not have to return. Here is just to make the API happy
        # Combine the channels
        rgbds = np.dstack([rgb, depth, semantic]).astype(np.uint8)
        return [rgbds]


def process(dataset_cfg, n_threads, n_parts, part_id, missing_dir, mode):
    """Process by part"""
    print('='*30 + '\n' + mode + '\n' + '='*30)
    _, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)

    # Create dataset
    dataset_params.update({
        'mode': mode,
        'modality': ['RGBDS'],
        'new_length': {'RGBDS': 1},
    })

    # Create the missing list by part
    missing_fname = os.path.join(missing_dir, 'missing.'+mode)
    assert os.path.isfile(missing_fname), \
        f'{missing_fname} not found'
    all_missing = open(missing_fname, 'r').read().splitlines()
    all_missing.sort()  # sort the fnames
    part_len = math.ceil(len(all_missing) / n_parts)
    part_missing = all_missing[part_id*part_len:(part_id+1)*part_len]

    dataset = EpicWrapper(n_parts, part_id, part_missing, **dataset_params)

    # Use torch DataLoader for parallel processing
    loader_params = {
        'batch_size': n_threads,
        'num_workers': n_threads,
        'pin_memory': False,
        'collate_fn': MiscUtils.safe_collate,  # safely remove broken samples
    }
    loader = DataLoader(dataset, shuffle=False, **loader_params)

    # Multithread processing
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
                        help='Dataset configuration')
    parser.add_argument('-m', '--missing_dir', type=str,
                        help='directory containing missing file names')
    args = parser.parse_args()

    assert 0 <= args.part_id < args.n_parts, \
        'part_id must be in [0, n_parts)'
    print('Part {} out of {}'.format(args.part_id+1, args.n_parts))

    assert os.path.isdir(args.missing_dir), \
        f'{args.missing_dir} not found'

    process(args.dataset_cfg, args.n_threads, args.n_parts, args.part_id,
            args.missing_dir, 'train')
    process(args.dataset_cfg, args.n_threads, args.n_parts, args.part_id,
            args.missing_dir, 'val')
    return 0


if __name__ == '__main__':
    sys.exit(main())
