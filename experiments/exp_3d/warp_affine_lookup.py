"""Find affine transformation using key points and warp

Look up the projection instead of actually running 3D->2D projection
"""
import os

import cv2
import numpy as np
import skimage.io as sio
from skimage.transform import resize
import matplotlib.pyplot as plt


def find_points_correspondence(ptid_1, ptid_2, pts_1, pts_2):
    """Find the point correspondence by looking at point id

    Args:
        ptid_1: (list) point id of the 1st frame
        ptid_2: (list) point id of the 2nd frame
        pts_1: (dict) point data of the 1st frame. The key has to match ptid_1
        pts_2: (dict) point data of the 2nd frame. The key has to match ptid_2

    Return:
        matched_1: (ndarray) matched points from the 1st frame
        matched_2: (ndarray) matched points from the 2nd frame
    """
    common_ids = np.intersect1d(ptid_1, ptid_2)
    matched_1 = np.array([pts_1[k] for k in common_ids], dtype=np.float32)
    matched_2 = np.array([pts_2[k] for k in common_ids], dtype=np.float32)
    return matched_1, matched_2


def read_inliner(path, fid):
    """Read inliner data

    Args:
        path: (str) path containing the data
        fid: (int) frame id

    Returns:
        ptid: (list) list of point id
        pt3d: (dict) dictionary of 3D locations
        pt2d: (dict) dictionary of 2D locations
    """
    fname = os.path.join(path, '0', 'PnPf',
                         'Inliers_{:04d}.txt'.format(fid))
    assert os.path.isfile(fname)

    content = open(fname).read().splitlines()
    n_pts = len(content)
    ptid = np.zeros(n_pts, dtype=np.int)
    pt3d = {}
    pt2d = {}

    for i, line in enumerate(content):
        tokens = line.split()
        pid, _, x, y, z, u, v, _ = tokens
        ptid[i] = int(pid)
        pt3d[ptid[i]] = np.array([float(x), float(y), float(z)], dtype=np.float32)
        pt2d[ptid[i]] = np.array([float(u), float(v)], dtype=np.float32)
    return ptid, pt3d, pt2d


def main():
    data_pth = '/home/knmac/Dropbox/SparseSensing/3d_projection/P01_08'
    frame_dir = '/home/knmac/projects/tmp_extract/frames_full/P01_08/0'

    frame1_id = 20
    frame2_id = 50
    frame1_pth = os.path.join(frame_dir, '{:04d}.jpg'.format(frame1_id))
    frame2_pth = os.path.join(frame_dir, '{:04d}.jpg'.format(frame2_id))

    # Read data ---------------------------------------------------------------
    # Read frames
    frame1 = sio.imread(frame1_pth)
    frame2 = sio.imread(frame2_pth)

    # Read inliners
    ptid_1, pt3d_1, pt2d_1 = read_inliner(data_pth, frame1_id)
    ptid_2, pt3d_2, pt2d_2 = read_inliner(data_pth, frame2_id)

    # Check 3D coordinates based on point id
    matched_1, matched_2 = find_points_correspondence(ptid_1, ptid_2, pt3d_1, pt3d_2)
    if np.all(matched_1 == matched_2):
        print('All common 3D coordinates match')

    # Match 2D coordinates ----------------------------------------------------
    matched_1, matched_2 = find_points_correspondence(ptid_1, ptid_2, pt2d_1, pt2d_2)

    # Warp the 1st frames to match the 2nd frame ------------------------------
    warp_mat, inliners = cv2.estimateAffine2D(matched_1, matched_2)
    warp_dst = cv2.warpAffine(frame1, warp_mat, (frame1.shape[1], frame1.shape[0]))

    # Distort the points for resized images -----------------------------------
    new_h, new_w = 224, 224
    frame1_resize = resize(frame1, (new_h, new_w), preserve_range=True,
                           anti_aliasing=True).astype(np.uint8)
    frame2_resize = resize(frame2, (new_h, new_w), preserve_range=True,
                           anti_aliasing=True).astype(np.uint8)
    scale_h = frame1.shape[0] / new_h
    scale_w = frame1.shape[1] / new_w
    matched_1_resize = matched_1 / [scale_w, scale_h]
    matched_2_resize = matched_2 / [scale_w, scale_h]
    warp_mat_resize, _ = cv2.estimateAffine2D(matched_1_resize, matched_2_resize)
    warp_dst_resize = cv2.warpAffine(frame1_resize, warp_mat_resize,
                                     (frame1_resize.shape[1], frame1_resize.shape[0]))

    # Visualize ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes[0, 0].imshow(frame1)
    axes[0, 1].imshow(frame2)
    axes[0, 2].imshow(warp_dst)

    axes[1, 0].imshow(frame1_resize)
    axes[1, 1].imshow(frame2_resize)
    axes[1, 2].imshow(warp_dst_resize)

    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Resized projected points')

    axes[1, 0].set_xlabel('Frame1')
    axes[1, 1].set_xlabel('Frame2')
    axes[1, 2].set_xlabel('Warped frame1 (to match frame2)')
    plt.show()


if __name__ == '__main__':
    main()
