import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

"""
本程序把transforms中的一些用得到的方法重写了，原因是图像分割在对图像img进行变换的时候同时要对结果mask进行变换，
原来的transforms输入输出的就是图片，而重写后的每个transforms的call方法输入输出的都是一个tuple，即img-target
"""


def pad_if_smaller(img, size, fill=0):
    """
    RadomCrop类使用此函数
    :param img:
    :param size:
    :param fill:
    :return:
    """
    # 如果图像最小边长小于给定size，则用数值fill进行padding
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
        """
        初始化：
        如果有最大尺寸则赋值，如果没有则最大最小尺寸都赋值为最小尺寸
        :param min_size: 最小尺寸
        :param max_size: 最大尺寸
        """
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        """
        :param image:tensor 将一个image重新定形（成一个正方形图片）resize
        :param target:tensor 将一个mask重新定形
        :return: tuple 图像目标对
        """
        # 从最小尺寸和最大尺寸中间选一个随机数
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        """
        :param flip_prob: float 图片翻转的概率
        """
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        # 产生一个随机数，如果小于flip_prob则直接翻转
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        """
        随机裁剪
        :param size: int 就是crop的值，裁剪大小的值
        """
        self.size = size

    def __call__(self, image, target):
        """
        就是一个随即裁剪的过程
        :param image:
        :param target:
        :return:
        """
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
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
