"""Test read 3d data using python and cpp
"""
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import time
import unittest
from subprocess import Popen, PIPE

import numpy as np

from src.utils.read_3d_data import (read_inliner, read_intrinsic_extrinsic,
                                    project_depth)


depth_path = 'data/EPIC_KITCHENS_2018/depth'
depth_tmpl = '0/PnPf/Inliers_{:04d}.txt'
vid = 'P01_08'
idx = 20
new_h, new_w = 224, 224

inliers_pth = os.path.join(depth_path, vid, depth_tmpl.format(idx))
corpus_pth = os.path.join(depth_path, vid, '0')
normalize_point_pth = os.path.join(depth_path, vid, '0', 'Points.txt')


def python_code():
    ptid, pt3d, pt2d = read_inliner(inliers_pth)

    frame_info = read_intrinsic_extrinsic(
        corpus_pth,
        startF=idx-1, stopF=idx-1,
    ).VideoInfo[idx-1]

    cam_center = frame_info.camCenter
    principle_ray_dir = frame_info.principleRayDir

    # Find depth wrt to camera coordinates
    depth, projection = project_depth(
        ptid, pt3d, pt2d, cam_center, principle_ray_dir,
        height=frame_info.height, width=frame_info.width,
        new_h=new_h, new_w=new_w,
    )

    # Normalize depth to the scale in milimeters
    assert os.path.isfile(normalize_point_pth)
    with open(normalize_point_pth, 'r') as fin:
        content = fin.read().splitlines()
    sfm_dist, real_dist = content[-1].split(' ')
    sfm_dist, real_dist = float(sfm_dist), float(real_dist)
    depth = depth / sfm_dist * real_dist

    return depth, projection


def cpp_code():
    # Get sfm_dist and real_dist to normalize
    assert os.path.isfile(normalize_point_pth)
    with open(normalize_point_pth, 'r') as fin:
        content = fin.read().splitlines()
    sfm_dist, real_dist = content[-1].split(' ')
    sfm_dist, real_dist = float(sfm_dist), float(real_dist)

    # Call bin file built from cpp
    cmd = [
        "./src/cpp_utils/read_3d_data/build/read_3d_data",
        inliers_pth,
        corpus_pth, "0", str(idx-1), str(idx-1), "224", "224", str(sfm_dist), str(real_dist),
    ]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    out, err = proc.communicate()

    # Parse output to python object
    tokens = out.split()
    depth = np.zeros([new_h, new_w])
    projection = {}
    for item in tokens:
        k, v, u, d = item.split(',')
        k, v, u, d = int(k), int(v), int(u), float(d)
        depth[v, u] = d
        projection[k] = (u, v)
    return depth, projection


class TestRead3D(unittest.TestCase):
    def test(self):
        st = time.time()
        depth1, projection1 = python_code()
        print('python code: {:.04f}s'.format(time.time() - st))

        st = time.time()
        depth2, projection2 = cpp_code()
        print('cpp code: {:.04f}s'.format(time.time() - st))

        assert (depth1 - depth2).max() < 1e-3
        assert projection1 == projection2


if __name__ == '__main__':
    unittest.main()
