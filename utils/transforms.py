"""
Taken from https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

torchvision.transforms.ToTensor should not be used when transforming target
image masks. This is PyTorch's reference implementation for image mask
transforms, plus some of my own custom transforms.
"""
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


# CUSTOM TRANSFORMS
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=Image.NEAREST)
        return image, target


class RandomCrop(object):
    """
    Set fill as an attribute and set default to zero.
    Fill should be any label id that gets converted into the ignore_id.
    """

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=self.fill)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomScaleCrop(object):
    """
    Randomly rescales the image and then applies a random crop.
    Fill should be any label id that gets converted into the ignore id.
    """

    def __init__(
        self,
        scale_min=0.5,
        scale_max=2.0,
        crop_size=512,
        inference_size=(512, 1024),  # height, width
        fill=0,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_size = crop_size
        self.inference_size = inference_size
        self.fill = fill

    def __call__(self, image, target):
        scale = random.uniform(self.scale_min, self.scale_max)
        height = int(self.inference_size[0] * scale)
        width = int(self.inference_size[1] * scale)
        size = (height, width)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)

        image = pad_if_smaller(image, self.crop_size, fill=0)
        target = pad_if_smaller(target, self.crop_size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.crop_size, self.crop_size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


# COPIED TRANSFORMS FROM PYTORCH
def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
