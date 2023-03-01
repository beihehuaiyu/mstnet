"""
数据预处理
"""

import collections
import numbers
import random

import numpy as np
import cv2
import scipy
import scipy.ndimage

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3, 4})


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + h, j:j + w]

def center_crop(img, output_size):
    if len(img.shape) == 3:
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        h, w = img.shape[0:2]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return crop(img, i, j, th, tw)
    elif len(img.shape) == 4:
        d, h, w = img.shape[:3]
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size), int(output_size))
        td, th, tw = output_size
        i = int(round((d - td) / 2.))
        j = int(round((h - th) / 2.))
        k = int(round((w - tw) / 2.))
        return crop3d(img, i, j, k, td, th, tw)

def crop3d(img, i, j, k, d, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + d, j:j + h, k:k+w, :]


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    h, w, c = img.shape[:3]

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        ow, oh = size[1], size[0]
    
    output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)

    if c == 1: output = output[..., None]

    return output