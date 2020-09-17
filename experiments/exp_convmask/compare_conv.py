"""Compare convolutions that can skips regions wrt a given mask
"""
import random
import time

import skimage.io as sio
from skimage.transform import resize
import numpy as np
import torch
from torch.nn import functional as F
from mmcv.ops import MaskedConv2d
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid", {'axes.grid': False})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def fix_seeds(seed=0):
    """Fix random seeds here for pytorch, numpy, and python"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def conv_standard(x, weight, padding):
    """Standard conv2d"""
    y = F.conv2d(x, weight, padding=padding)
    return y


def conv_for(x, weight, padding=(1, 1), dilation=(1, 1), stride=(1, 1)):
    """Naive implementation of Conv2d using for loop"""
    b_s, _, in_h, in_w = x.shape
    out_c, in_c, k_h, k_w = weight.shape
    pad_h, pad_w = padding
    d_h, d_w = dilation

    out_w = (in_w + 2*pad_w - k_w) // d_w + 1
    out_h = (in_h + 2*pad_h - k_h) // d_h + 1
    y = torch.zeros((b_s, out_c, out_h, out_w), dtype=torch.float32).to(device)
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'constant', 0)

    u = 0
    for i in range(out_h):
        v = 0
        for j in range(out_w):
            patch = x_pad[:, :,
                          pad_h+u-k_h//2: pad_h+u+k_h//2+1,
                          pad_w+v-k_w//2: pad_w+v+k_w//2+1]
            y[:, :, i, j] = (patch * weight).sum(dim=[1, 2, 3])
            v += stride[0]
        u += stride[1]
    return y


def conv_fold(x, weight, padding, dilation=(1, 1)):
    """Conv with fold"""
    b_s, _, in_h, in_w = x.shape
    out_c, in_c, k_h, k_w = weight.shape
    pad_h, pad_w = padding
    d_h, d_w = dilation

    out_w = (in_w + 2*pad_w - k_w) // d_w + 1
    out_h = (in_h + 2*pad_h - k_h) // d_h + 1

    x_unf = F.unfold(x, kernel_size=(k_h, k_w), padding=padding)
    y_unf = x_unf.transpose(1, 2).matmul(weight.view(out_c, -1).t()).transpose(1, 2)
    # y = F.fold(y_unf, (out_h, out_w), (1, 1))
    y = y_unf.view(b_s, out_c, out_h, out_w)
    return y


def conv_mask_input(x, weight, mask, padding):
    """Standard conv2d with masking input"""
    y = F.conv2d(mask * x, weight, padding=padding)
    return y


def conv_mask_output(x, weight, mask, padding):
    """Standard conv2d with masking output"""
    y = F.conv2d(x, weight, padding=padding)
    return y * mask


def conv_mask_for(x, weight, mask, padding=(1, 1), dilation=(1, 1), stride=(1, 1)):
    """Naive implementation of Conv2d with masking using for loop"""
    b_s, _, in_h, in_w = x.shape
    out_c, in_c, k_h, k_w = weight.shape
    pad_h, pad_w = padding
    d_h, d_w = dilation

    out_w = (in_w + 2*pad_w - k_w) // d_w + 1
    out_h = (in_h + 2*pad_h - k_h) // d_h + 1
    y = torch.zeros((b_s, out_c, out_h, out_w), dtype=torch.float32).to(device)
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'constant', 0)

    u = 0
    for i in range(out_h):
        v = 0
        for j in range(out_w):
            if mask[0, i, j] == 0:
                v += stride[0]
                continue
            patch = x_pad[:, :,
                          pad_h+u-k_h//2: pad_h+u+k_h//2+1,
                          pad_w+v-k_w//2: pad_w+v+k_w//2+1]
            y[:, :, i, j] = (patch * weight).sum(dim=[1, 2, 3])
            v += stride[0]
        u += stride[1]
    return y


def conv_mask_fold(x, weight, mask, padding, dilation=(1, 1)):
    """Conv with fold and mask"""
    b_s, _, in_h, in_w = x.shape
    out_c, in_c, k_h, k_w = weight.shape
    pad_h, pad_w = padding
    d_h, d_w = dilation

    out_w = (in_w + 2*pad_w - k_w) // d_w + 1
    out_h = (in_h + 2*pad_h - k_h) // d_h + 1

    x_unf = F.unfold(x, kernel_size=(k_h, k_w), padding=padding)
    mask_flat = mask.flatten()
    idx = torch.where(mask_flat == 1)[0]
    x_sel = x_unf[:, :, idx]
    y_sel = x_sel.transpose(1, 2).matmul(weight.view(out_c, -1).t()).transpose(1, 2)

    y_unf = torch.zeros([b_s, out_c, out_h*out_w], dtype=torch.float32).to(device)
    y_unf[:, :, idx] = y_sel
    # y_unf = x_unf.transpose(1, 2).matmul(weight.view(out_c, -1).t()).transpose(1, 2)
    # y = F.fold(y_unf, (out_h, out_w), (1, 1))
    y = y_unf.view(b_s, out_c, out_h, out_w)
    return y


def main():
    H, W = 224, 224
    # H, W = 14, 14
    in_c, out_c = 3, 10
    k_size = 3
    padding = (k_size//2, k_size//2)

    # Read image
    img = sio.imread('./baboon.jpg')
    img = resize(img, [H, W], anti_aliasing=True).astype(np.float32)

    # Prepare variables
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(dim=0).to(device).contiguous()
    # x = torch.rand(1, 3, 224, 224).cuda()
    mask = torch.zeros([1, H, W]).to(device)
    mask[:, H//4: 3*H//4, W//4: 3*W//4] = 1.0
    weight = torch.rand([out_c, in_c, k_size, k_size]).to(device) - 0.5

    mask_conv = MaskedConv2d(in_c, out_c, k_size, stride=1, padding=padding[0]).to(device)
    mask_conv.weight.data = weight.clone().detach()
    mask_conv.bias.data = torch.zeros(mask_conv.bias.shape).to(device)

    # Warm up gpu and caching
    print('Warming up...')
    for _ in range(3):
        tmp = conv_standard(x, weight, padding)
        conv_for(x, weight, padding)
        conv_fold(x, weight, padding)
        conv_mask_input(x, weight, mask, padding)
        conv_mask_output(x, weight, mask, padding)
        conv_mask_for(x, weight, mask, padding)
        conv_mask_fold(x, weight, mask, padding)
        mask_conv(x, mask)

        vmin = tmp[0].mean(dim=0).min().item()
        vmax = tmp[0].mean(dim=0).max().item()

    # Run experiments multiple times
    res_dict = {
        't0': [], 't1': [], 't2': [],
        't3': [], 't4': [], 't5': [], 't6': [], 't7': [],
    }

    for i in range(10):
        print('--------------------------------------------------------------')
        print('run', i)
        # -------------------------------------------------------------------------
        # Experiments with conv2d
        print('Experiments with conv2d')
        st = time.time()
        y0 = conv_standard(x, weight, padding)
        t0 = time.time() - st
        print('- Standard conv: {:.06f}s'.format(t0))

        st = time.time()
        y1 = conv_for(x, weight, padding)
        t1 = time.time() - st
        print('- For loop conv: {:.06f}s'.format(t1))

        st = time.time()
        y2 = conv_fold(x, weight, padding)
        t2 = time.time() - st
        print('- Fold conv:     {:.06f}s'.format(t2))

        # -------------------------------------------------------------------------
        # Experiments with conv2d and mask
        print('Experiments with conv2d and mask')
        st = time.time()
        y3 = conv_mask_input(x, weight, mask, padding)
        t3 = time.time() - st
        print('- Mask input:         {:.06f}s'.format(t3))

        st = time.time()
        y4 = conv_mask_output(x, weight, mask, padding)
        t4 = time.time() - st
        print('- Mask output:        {:.06f}s'.format(t4))

        st = time.time()
        y5 = conv_mask_for(x, weight, mask, padding)
        t5 = time.time() - st
        print('- Mask for loop conv: {:.06f}s'.format(t5))

        st = time.time()
        y6 = conv_mask_fold(x, weight, mask, padding)
        t6 = time.time() - st
        print('- Mask fold conv:     {:.06f}s'.format(t6))

        st = time.time()
        y7 = mask_conv(x, mask)
        t7 = time.time() - st
        print('- MaskedConv2d:       {:.06f}s'.format(t7))

        res_dict['t0'].append(t0)
        res_dict['t1'].append(t1)
        res_dict['t2'].append(t2)
        res_dict['t3'].append(t3)
        res_dict['t4'].append(t4)
        res_dict['t5'].append(t5)
        res_dict['t6'].append(t6)
        res_dict['t7'].append(t7)

    # -------------------------------------------------------------------------
    for k in res_dict:
        res_dict[k] = np.array(res_dict[k])

    print('==================================================================')
    print('  Overall:')
    print('Experiments with conv2d')
    print('- Standard conv: {:.06f} +- {:.06f}s'.format(
        res_dict['t0'].mean(), res_dict['t0'].std()))
    print('- For loop conv: {:.06f} +- {:.06f}s'.format(
        res_dict['t1'].mean(), res_dict['t1'].std()))
    print('- Fold conv:     {:.06f} +- {:.06f}s'.format(
        res_dict['t2'].mean(), res_dict['t2'].std()))

    print('Experiments with conv2d and mask')
    print('- Mask input:         {:.06f} +- {:.06f}s'.format(
        res_dict['t3'].mean(), res_dict['t3'].std()))
    print('- Mask output:        {:.06f} +- {:.06f}s'.format(
        res_dict['t4'].mean(), res_dict['t4'].std()))
    print('- Mask for loop conv: {:.06f} +- {:.06f}s'.format(
        res_dict['t5'].mean(), res_dict['t5'].std()))
    print('- Mask fold conv:     {:.06f} +- {:.06f}s'.format(
        res_dict['t6'].mean(), res_dict['t6'].std()))
    print('- MaskedConv2d:       {:.06f} +- {:.06f}s'.format(
        res_dict['t7'].mean(), res_dict['t7'].std()))

    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))
    axes[0].imshow(y0[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[1].imshow(y1[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Max diff={:.2e}'.format((y1-y0).abs().max().item()))
    axes[2].imshow(y2[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('Max diff={:.2e}'.format((y2-y0).abs().max().item()))

    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 3))
    axes[0].imshow(y3[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[1].imshow(y4[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[2].imshow(y5[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('Max diff={:.2e}'.format((y5-y4).abs().max().item()))
    axes[3].imshow(y6[0].mean(dim=0).cpu().numpy(), vmin=vmin, vmax=vmax)
    axes[3].set_xlabel('Max diff={:.2e}'.format((y6-y4).abs().max().item()))
    axes[4].imshow(y7[0].mean(dim=0).cpu().detach().numpy(), vmin=vmin, vmax=vmax)
    axes[4].set_xlabel('Max diff={:.2e}'.format((y7-y4).abs().max().item()))
    plt.show()


if __name__ == '__main__':
    fix_seeds()
    main()
