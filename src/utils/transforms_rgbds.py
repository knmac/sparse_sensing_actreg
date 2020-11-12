"""Tensor transformation for RGBDS data

Reimplementation of necessary functions intransforms.py, but using numpy array
inputs instead of PIL Image
"""
import random
from skimage.transform import resize
import numpy as np
import torch


class GroupCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        img_h = img_group[0].shape[0]
        img_w = img_group[0].shape[1]

        top = (img_h - self.size) // 2
        left = (img_w - self.size) // 2

        ret = [img[top:top+self.size, left:left+self.size, :] for img in img_group]
        return ret


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img[:, ::-1, :] for img in img_group]
            return ret
        else:
            return img_group


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        img_h = img_group[0].shape[0]
        img_w = img_group[0].shape[1]

        if img_h > img_w:
            ret = [resize(img, (int(self.size * img_h / img_w), self.size),
                          preserve_range=True, anti_aliasing=True)
                   for img in img_group]
        else:
            ret = [resize(img, (self.size, int(self.size * img_w / img_h)),
                          preserve_range=True, anti_aliasing=True)
                   for img in img_group]
        return ret


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True,
                 more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (input_size if not isinstance(input_size, int)
                           else [input_size, input_size])

    def __call__(self, img_group):
        im_shape = img_group[0].shape

        crop_h, crop_w, offset_h, offset_w = self._sample_crop_size(
            image_h=im_shape[0], image_w=im_shape[1])
        crop_img_group = [img[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]
                          for img in img_group]
        ret_img_group = [resize(img, self.input_size, preserve_range=True, anti_aliasing=True)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, image_h, image_w):
        # find a crop size
        base_size = min(image_h, image_w)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[0] if abs(x-self.input_size[0]) < 3 else x
                  for x in crop_sizes]
        crop_w = [self.input_size[1] if abs(x-self.input_size[1]) < 3 else x
                  for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((h, w))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            h_offset = random.randint(0, image_h - crop_pair[0])
            w_offset = random.randint(0, image_w - crop_pair[1])
        else:
            h_offset, w_offset = self._sample_fix_offset(
                image_h, image_w, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], int(h_offset), int(w_offset)

    def _sample_fix_offset(self, image_h, image_w, crop_h, crop_w):
        offsets = self.fill_fix_offset(self.more_fix_crop,
                                       image_h, image_w, crop_h, crop_w)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_h, image_w, crop_h, crop_w):
        h_step = (image_h - crop_h) / 4
        w_step = (image_w - crop_w) / 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((0, 4 * w_step))  # upper right
        ret.append((4 * h_step, 0))  # lower left
        ret.append((4 * h_step, 4 * w_step))  # lower right
        ret.append((2 * h_step, 2 * w_step))  # center

        if more_fix_crop:
            ret.append((2 * h_step, 0))  # center left
            ret.append((2 * h_step, 4 * w_step))  # center right
            ret.append((4 * h_step, 2 * w_step))  # lower center
            ret.append((0 * h_step, 2 * w_step))  # upper center

            ret.append((1 * h_step, 1 * w_step))  # upper left quarter
            ret.append((1 * h_step, 3 * w_step))  # upper right quarter
            ret.append((3 * h_step, 1 * w_step))  # lower left quarter
            ret.append((3 * h_step, 3 * w_step))  # lower righ quarter
        return ret


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if self.roll:
            return np.concatenate([x[:, :, ::-1] for x in img_group], axis=2)
        return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W)"""
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)


class IdentityTransform(object):

    def __call__(self, data):
        return data
