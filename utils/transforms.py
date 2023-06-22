import os
import random
import sys
import platform
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from utils.handy_functions import double_img_show, format_convert

sys.path.append("..")

"""
本程序把transforms中的一些用得到的方法重写了，原因是图像分割在对图像img进行变换的时候同时要对结果mask进行变换，
原来的transforms输入输出的就是一张图片，而重写后的每个transforms的call方法输入输出的都是两张图片，即img-target
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
        """
        用于组合transforms
        :param transforms: transforms类
        """
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
        将一组输入-目标按照生成的随机数进行放大或缩小，但图片的长宽比不变，随机的范围由上述max_size和min_size界定
        """
        # 从最小尺寸和最大尺寸中间选一个随机数
        size = random.randint(self.min_size, self.max_size)
        # 调试时候打印一下出来的size是多少，之后这行注释掉
        # print('The radom size is:' + str(size))
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, size):
        """
        初始化：
        """
        self.size = size

    def __call__(self, image, target):
        """
        :param image:tensor 将一个image重新定形（成一个正方形图片）resize
        :param target:tensor 将一个mask重新定形
        :return: tuple 图像目标对
        """
        size = self.size
        # 调试时候打印一下出来的size是多少，之后这行注释掉
        # print('The radom size is:' + str(size))
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


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
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
        # 我感觉他写错了，应该填0黑色才对，但他在处理的时候进行了一个颜色反转，所以还是255
        target = pad_if_smaller(target, self.size, fill=255)
        # target = pad_if_smaller(target, self.size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        """
        以中心进行裁剪，输出的图片是正方形size*size
        :param size:
        """
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


if __name__ == "__main__":
    # 调试代码使用
    # 图片路径
    # pic_path = "../test_picture"
    # pic_path = "E:\\Datasets\\CHASE_DB1"
    pic_path = "E:\\Datasets\\ISIC2018\\ISIC2018_Task1-2_Training_Input"
    # 图片候选，用哪个去掉注释即可
    # input_pic_name = "Image_01L.jpg"
    input_pic_name = "ISIC_0000080.jpg"
    # pic_name = "01_test.tif"
    # pic_name = "ILSVRC2012_test_00000038.png"
    input_pic_path = os.path.join(pic_path, input_pic_name)
    # target_pic_name = "Image_01L_1stHO.png"
    target_pic_name = "ISIC_0000080_segmentation.png"
    # target_pic_path = os.path.join(pic_path, target_pic_name)
    pic_path2 = "E:\\Datasets\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth"
    target_pic_path = os.path.join(pic_path2, target_pic_name)

    # # 使用CV2测试一下输出
    # # flags 1是rgb图 0是灰度图 -1是原图
    # input_pic_cv2 = cv2.imread(input_pic_path, flags=1)
    # cv2.imshow('flag=1 rgb pic', input_pic_cv2)
    # input_pic_cv2 = cv2.imread(input_pic_path, flags=0)
    # cv2.imshow('flag=0 gray pic', input_pic_cv2)
    # input_pic_cv2 = cv2.imread(input_pic_path, flags=-1)
    # cv2.imshow('flag=-1 origin pic', input_pic_cv2)
    # # 等待窗口关闭或任意键输入继续
    # cv2.waitKey(0)

    # 先放原图
    # input_pic = cv2.imread(input_pic_path)
    # target_pic = cv2.imread(target_pic_path)
    # if platform.system() == "Windows":
    #     cv2.imshow('original input picture', input_pic)
    #     cv2.imshow('original target picture', target_pic)
    #     cv2.waitKey(0)

    judge = int(input("Input a number\n"
                      "1:RandomResize\n"
                      "2:RandomHorizontalFlip\n"
                      "3:RandomCrop\n"
                      "4:CenterCrop\n"
                      "5:ToTensor&Normalize\n"
                      "6:Resize"
                      ":"))
    input_pic = Image.open(input_pic_path)
    target_pic = Image.open(target_pic_path)

    if judge == 1:
        # 测试RandomResize，注意此处F.resize必须要PIL类的图片
        # 开始随机resize
        size = 960
        print("Before random,\ninput picture size:", input_pic.size, "\ntarget picture size:", target_pic.size)
        input_pic, target_pic = RandomResize(int(0.5 * size), int(1.2 * size))(input_pic, target_pic)
        print("After random,\ninput picture size:", input_pic.size, "\ntarget picture size:", target_pic.size)
    elif judge == 2:
        # 测试RandomHorizontalFlip
        print("Do horizontal flip:")
        input_pic, target_pic = RandomHorizontalFlip(1)(input_pic, target_pic)
    elif judge == 3:
        # 测试RandomCrop
        size = 1200
        print("Do random crop...")
        input_pic, target_pic = RandomCrop(size)(input_pic, target_pic)
        print("After random,\ninput picture size:", input_pic.size, "\ntarget picture size:", target_pic.size)
    elif judge == 4:
        # 测试CenterCrop
        size = 1200
        print("Do center crop...")
        input_pic, target_pic = CenterCrop(size)(input_pic, target_pic)
        print("After center,\ninput picture size:", input_pic.size, "\ntarget picture size:", target_pic.size)
    elif judge == 5:
        # 测试ToTensor和Normalize
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        print("Changing to tensor...")
        input_pic, target_pic = ToTensor()(input_pic, target_pic)
        # 将其转化成float的tensor，主要是mask，否则mask是int形式的01
        input_pic = input_pic.float()
        target_pic = target_pic.float()
        # unsqueeze是增添一维，后面的数字在于在哪维增这维
        target_pic = torch.unsqueeze(target_pic, 0)
        print("After to tensor,\ninput picture size:", input_pic.shape, "\ntarget picture size:", target_pic.shape)
        print("Do normalization...")
        input_pic, target_pic = Normalize(mean, std)(input_pic, target_pic)
        print("After normalization,\ninput picture size:", input_pic.shape, "\ntarget picture size:", target_pic.shape)
        # 这里注意ToPILImage是C*H*W的Tensor或者H*W*C的numpy转成PIL
        # ToTensor是H*W*C的numpy转成tensor，如果是PIL进去或者满足np.uint8进去则是正常的C*H*W
        # input_pic, = T.ToPILImage()(input_pic)
        # target_pic = T.ToPILImage()(target_pic)
    elif judge == 6:
        # 测试RandomResize，注意此处F.resize必须要PIL类的图片
        # 开始随机resize
        size = 512
        print("Before resize,\ninput picture size:", input_pic.size, "\ntarget picture size:", target_pic.size)
        input_pic, target_pic = Resize(size)(input_pic, target_pic)
        print("After resize,\ninput picture size:", input_pic.size, "\ntarget picture size:", target_pic.size)

    if judge != 5 and platform.system() == "Windows":
        # 输出图片效果，注意5是不行的
        # 将PIL的视频转化为numpy，才能使用CV2输出，而且注意，cv2存储的图片需要先cvtColor一下才能正常输出，因为cv2输出时是BGR输出
        # input_pic = cv2.cvtColor(np.array(input_pic), cv2.COLOR_BGR2RGB)
        # target_pic = cv2.cvtColor(np.array(target_pic), cv2.COLOR_BGR2RGB)
        # cv2.imshow('after process input pic', input_pic)
        # cv2.imshow('after process target pic', target_pic)
        # cv2.waitKey(0)
        double_img_show(format_convert(input_pic), format_convert(target_pic))
