"""Make RBF interpolation from projection
"""
import os
import time

import numpy as np
import skimage.io as sio
from skimage.transform import resize
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import seaborn as sns

from warp_affine_lookup import read_inliner
from read_3d_data import read_corpus, read_intrinsic_extrinsic


def make_proj_img(ptid, pt3d, pt2d, cam_center, principle_ray_dir,
                  height, width, new_h, new_w, rbf_opts):
    scale_h = height / new_h
    scale_w = width / new_w

    # Collect and rescale 2d points
    img = np.zeros([new_h, new_w])
    for k in ptid:
        u, v = np.round(pt2d[k] / [scale_w, scale_h]).astype(int)
        if u < 0 or u >= new_w or v < 0 or v >= new_h:
            continue

        cam2point = pt3d[k] - cam_center
        depth = np.dot(cam2point, principle_ray_dir)
        img[v, u] = depth

    # Get the unique points
    nz_idx = np.where(img > 0)
    y, x = nz_idx
    z = img[nz_idx]

    # Interpolation
    # rbf = Rbf(x, y, z, epsilon=epsilon, function='gaussian')
    rbf = Rbf(x, y, z, **rbf_opts)
    yy, xx = np.meshgrid(np.arange(new_h), np.arange(new_w))
    zz = rbf(yy, xx)
    return (x, y, z), (xx, yy, zz)


def main():
    data_pth = '/home/knmac/Dropbox/SparseSensing/3d_projection/P01_08'
    frame_dir = '/home/knmac/projects/tmp_extract/frames_full/P01_08/0'
    frame_id = 20
    new_h, new_w = 112, 112

    # Read data ---------------------------------------------------------------
    # Read image
    frame_pth = os.path.join(frame_dir, '{:04d}.jpg'.format(frame_id))
    frame = sio.imread(frame_pth)
    frame_resize = resize(frame, (new_h, new_w), preserve_range=True,
                          anti_aliasing=True).astype(np.uint8)

    # Read corpus
    corpus_info, vcorpus_cid_lcid_lfid = read_corpus(data_pth)

    # Read camera parameters
    vinfo = read_intrinsic_extrinsic(data_pth)

    # Read inliners
    ptid, pt3d, pt2d = read_inliner(data_pth, frame_id)

    # Process projection ------------------------------------------------------
    cam_center = vinfo.VideoInfo[frame_id].camCenter
    principle_ray_dir = vinfo.VideoInfo[frame_id].principleRayDir
    rbf_opts = {
        'function': 'linear',
        'epsilon': 2.0,
    }
    foo, bar = make_proj_img(ptid, pt3d, pt2d, cam_center, principle_ray_dir,
                             height=1080, width=1920, new_h=new_h, new_w=new_w,
                             rbf_opts=rbf_opts)
    x, y, z = foo
    xx, yy, zz = bar

    # Visualize ---------------------------------------------------------------
    sns.set_style("whitegrid", {'axes.grid': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frame_resize)
    axes[1].scatter(x, y, 50, z)
    # axes[1].pcolor(xx, yy, zz)
    axes[2].imshow(zz)

    axes[1].set_ylim(axes[1].get_ylim()[::-1])
    axes[1].axis('equal')

    axes[0].set_title('RGB')
    axes[1].set_title('Projection')
    axes[2].set_title('RBF Interpolation')
    plt.show()

    # Check runtime -----------------------------------------------------------
    for func in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
                 'quintic', 'thin_plate']:
        rbf_opts = {'function': func, 'epsilon': 2.0}
        st = time.time()
        foo, bar = make_proj_img(ptid, pt3d, pt2d, cam_center, principle_ray_dir,
                                 height=1080, width=1920, new_h=new_h, new_w=new_w,
                                 rbf_opts=rbf_opts)
        print(func, '->', time.time()-st)


if __name__ == '__main__':
    main()
