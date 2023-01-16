import os

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import init
from torchvision import transforms


def val_range(input_val, name='default name'):
    """
    返回一个图像/numpy/tensor的最大值最小值，在处理过程中打印其尺寸
    :param name:该输入的名字
    :param input_val:输入
    :return:最小值和最大值
    """
    print("------------------------------------------")
    if type(input_val) == torch.Tensor:
        min_val = torch.min(input_val)
        max_val = torch.max(input_val)
        print("Tensor:", name,
              "\nhas a shape of", input_val.shape,
              "\nhas a range between", min_val, "and", max_val)
    elif type(input_val) == np.ndarray:
        min_val = np.min(input_val)
        max_val = np.max(input_val)
        print("Numpy array:", name,
              "\nhas a shape of", input_val.shape,
              "\nhas a range between", min_val, "and", max_val)
    else:
        input_val = np.array(input_val)
        min_val = np.min(input_val)
        max_val = np.max(input_val)
        print("Other (PIL Image):", name,
              "\nhas a shape of", input_val.shape,
              "\nhas a range between", min_val, "and", max_val)
    print("------------------------------------------")
    return min_val, max_val


def img_show(img):
    """
    使用matplotlib展示图片
    :param img:图片
    """
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()


def double_img_show(input_img, predicted_img):
    """
    用于展示原图+预测结果
    :param input_img: 输入图片
    :param predicted_img: 预测结果
    """
    plt.subplot(121)
    # 不显示坐标轴
    plt.axis('off')
    plt.imshow(input_img, cmap='gray')
    plt.title('input image')

    plt.subplot(122)
    plt.axis('off')
    plt.imshow(predicted_img, cmap='gray')
    plt.title('prediction')

    plt.show()


def triple_img_show(original_img, original_mask, predicted_img):
    """
    用于展示原图，标准答案，预测结果
    :param original_img: 原图
    :param original_mask: 原mask
    :param predicted_img: 预测结果
    """
    plt.subplot(131)
    # 不显示坐标轴
    plt.axis('off')
    plt.imshow(original_img, cmap='gray')
    plt.title('original image')

    plt.subplot(132)
    plt.axis('off')
    plt.imshow(original_mask, cmap='gray')
    plt.title('original mask')

    plt.subplot(133)
    plt.axis('off')
    plt.imshow(predicted_img, cmap='gray')
    plt.title('prediction')

    plt.show()


def format_convert(input_format):
    """
    :param input: 可能是tensor，可能是PIL，可能是Numpy
    :return: 一张PIL图片
    """
    pil_output = None
    if type(input_format) == torch.Tensor:
        if (input_format.shape.__len__() == 4 and input_format.shape[0] == 1) \
                or (input_format.shape.__len__() == 3 and input_format.shape[0] == 1):
            # 证明这个带了batchsize
            input_format = input_format.squeeze(0)
            input_format = input_format.type(torch.uint8)
            pil_output = transforms.ToPILImage()(input_format)
            pil_output = pil_output.convert('RGB')
        elif ((input_format.shape[0] == 3 or input_format.shape[0] == 4) and input_format.shape.__len__() == 3) \
                or input_format.shape.__len__() == 2:
            pil_output = transforms.ToPILImage()(input_format)
            pil_output = pil_output.convert('RGB')
        else:
            raise TypeError("The input is not a picture.")
    elif type(input_format) == np.ndarray:
        pil_output = PIL.Image.fromarray(np.uint8(input_format))
        pil_output = pil_output.convert('RGB')
    else:
        pil_output = input_format
        pil_output = pil_output.convert('RGB')
    return pil_output


# init_weights使用方法
# net = Net(params...)
# init_weights(net)
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")

    for item in os.listdir(path):
        if '.git' not in item:
            print("|      " * depth + "+--" + item)
            newitem = path + '/' + item
            if os.path.isdir(newitem):
                dfs_showdir(newitem, depth + 1)


def channel_extract(pic_path, channel, save_dir='', save_name=''):
    """
    :param pic_path: 图片文件路径，一张图片，如果是灰度图，会转成三通道RGB后提取对应通道
    :param channel: 要提取出哪个通道
    :return:
    """
    if channel not in ['r', 'g', 'b', 'R', 'G', 'B']:
        raise ValueError('channel must be R/G/B')

    pic = PIL.Image.open(pic_path).convert('RGB')
    pic_tensor = transforms.ToTensor()(pic)
    # 由于要清空其它通道，所以r对应1和2
    value_dict = {'r': [1, 2], 'R': [1, 2], 'g': [0, 2], 'G': [0, 2], 'b': [0, 1], 'B': [0, 1]}
    for i in value_dict[channel]:
        pic_tensor[i, :, :] = 0.0
    result = format_convert(pic_tensor)

    # 显示图片
    # img_show(result)

    # 保存图片
    save_path = save_file_generate(original_dir=pic_path, save_dir=save_dir, save_name=save_name)
    result.save(save_path)


def save_file_generate(original_dir='', save_dir='', save_name=''):
    """
    :param original_dir: 原来的文件是什么
    :param save_dir: 存在哪个文件夹下
    :param save_name: 存的名字是什么，如果非空则要带后缀
    :return:一个合理的路径
    """
    if original_dir == '':
        # 新诞生的文件，此时save_name不能为空
        if save_name == '' or save_dir == '':
            raise ValueError('Path or file name cannot be empty.')
        else:
            return os.path.join(os.path.abspath(save_dir), save_name)
    else:
        # 原来就有的文件
        original_dir = os.path.abspath(original_dir)
        if save_dir == '':
            save_dir = os.path.dirname(original_dir)
        else:
            save_dir = os.path.abspath(save_dir)

        # 如果给到的文件名是空，那么以原名_new来命名
        if save_name == '':
            # 获取后缀
            suffix = os.path.splitext(original_dir)[-1]
            # 获取原来文件名
            original_name = os.path.basename(original_dir).split('.')[0]
            save_name = original_name + '_new' + suffix

        return os.path.join(os.path.abspath(save_dir), save_name)


if __name__ == '__main__':
    # pic_path = 'E:/Datasets/DRIVE/test/2nd_manual/01_manual2.gif'
    pic_path = 'E:/Datasets/DRIVE/test/images/01_test.tif'
    newpath = save_file_generate(original_dir=pic_path)

    channel_extract(pic_path, 'r', save_name='r.tif')
    channel_extract(pic_path, 'g', save_name='g.tif')
    channel_extract(pic_path, 'b', save_name='b.tif')
    pass
