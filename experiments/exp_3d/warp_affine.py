"""Find affine transformation using key points and warp
"""
import os

import cv2
import numpy as np
import skimage.io as sio
from skimage.transform import resize
import matplotlib.pyplot as plt

from read_3d_data import (read_corpus,
                          read_intrinsic_extrinsic,
                          project_frame)


def find_points_correspondence(pt2d_1, pt3d_1, pt2d_2, pt3d_2):
    """
    Args:
        pt2d_1: 2D points in image plane from the first frame
        pt3d_1: 3D points in world plane from the first frame
        pt2d_2: 2D points in image plane from the second frame
        pt3d_2: 3D points in world plane from the second frame

    Return:
        set1: matched 2D points from the first frame
        set2: matched 2D points from the second frame
    """
    n1, n2 = len(pt2d_1), len(pt2d_2)
    assert len(pt3d_1) == n1 and len(pt3d_2) == n2

    # Matching the indices by looing at the 3D points
    matching = np.ones(n1, dtype=int) * -1
    taken = np.zeros(n2, dtype=np.uint8)
    for i in range(n1):
        # Find the index of the 3D point in the second frame
        indices = np.where((pt3d_2 == pt3d_1[i]).all(axis=1))[0]
        if len(indices) == 0:  # No matching
            continue
        elif len(indices) == 1:  # One matching
            idx = indices[0]
            matching[i] = idx
            taken[idx] = 1
        else:  # Multiple matchings
            for idx in indices:
                if taken[idx] == 1:
                    continue
                else:
                    matching[i] = idx
                    taken[idx] = 1
                    break

    # Collect the matched 2D points
    set1, set2 = [], []
    for i in range(n1):
        if matching[i] == -1:
            continue
        set1.append(pt2d_1[i])
        set2.append(pt2d_2[matching[i]])
    set1 = np.array(set1)
    set2 = np.array(set2)

    return set1, set2


# def find_essential(pts1, pts2):
#     """Find the essential matrix (or fundamental matrix) using eight-point algorithm

#     Args:
#         pts1: (N, 2) array. 2D points of the first frame
#         pts2: (N, 2) array. 2D points of the second frame

#     Ref:
#         https://en.wikipedia.org/wiki/Eight-point_algorithm
#     """
#     assert len(pts1) == len(pts2)
#     N = len(pts1)

#     # Step 1: Fomulating a homogeneous linear equation
#     Y = np.zeros((9, N), dtype=float)
#     for k in range(N):
#         y1, y2 = pts1[k]
#         y1_, y2_ = pts2[k]
#         y_tilde_k = [y1_*y1, y1_*y2, y1_, y2_*y1, y2_*y2, y2_, y1, y2, 1.0]
#         Y[:, k] = y_tilde_k

#     # Step 2: Solving the equation
#     U, S, Vh = np.linalg.svd(Y)
#     assert S[-1] == S.min()
#     # Last column = left singular vector of smallest singular value
#     E_est = U[:, -1].reshape(3, 3)

#     # Step 3: Enforcing the internal constraint
#     U, S, Vh = np.linalg.svd(E_est)
#     s1, s2 = S[0], S[1]
#     S_ = np.diag([s1, s2, 0.])
#     E = np.matmul(np.matmul(U, S_), Vh)
#     return E/E[2, 2]


def main():
    data_pth = '/home/knmac/Dropbox/SparseSensing/3d_projection/P01_08'
    frame_dir = '/home/knmac/projects/tmp_extract/frames_full/P01_08/0'
    frame1_id = 20
    frame2_id = 50
    frame1_pth = os.path.join(frame_dir, '{:04d}.jpg'.format(frame1_id))
    frame2_pth = os.path.join(frame_dir, '{:04d}.jpg'.format(frame2_id))

    # Read data ---------------------------------------------------------------
    # Read corpus
    corpus_info, vcorpus_cid_lcid_lfid = read_corpus(data_pth)

    # Read camera parameters
    vinfo = read_intrinsic_extrinsic(data_pth)

    # Read frames
    frame1 = sio.imread(frame1_pth)
    frame2 = sio.imread(frame2_pth)

    # Match and warp ----------------------------------------------------------
    pt2d_1, _, _, pt3d_1 = project_frame(
        frame1_id, vinfo, corpus_info, vcorpus_cid_lcid_lfid, unique_points=True)
    pt2d_2, _, _, pt3d_2 = project_frame(
        frame2_id, vinfo, corpus_info, vcorpus_cid_lcid_lfid, unique_points=True)

    matched_1, matched_2 = find_points_correspondence(pt2d_1, pt3d_1, pt2d_2, pt3d_2)
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
    return 0


if __name__ == '__main__':
    main()
