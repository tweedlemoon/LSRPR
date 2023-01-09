import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms

from utils.handy_functions import img_show, val_range

# 预测调色盘
# 前面是ground truth，后面是prediction
# 0,0->0设为黑
# 意思是ground truth该像素点为0，预测结果也是0
# 1,0->1设为蓝
# 0,1->2设为红
# 1,1->3设为绿
VESSEL_PALETTE = np.asarray(
    [
        # 黑
        [0, 0, 0],
        # 蓝
        [0, 0, 255],
        # 红
        [255, 0, 0],
        # 绿
        [0, 255, 0],
    ], dtype=np.uint8
)


def color_img(prediction, file_name='', palette=VESSEL_PALETTE):
    '''
    :param prediction: tensor[1,W,H]，输入的单张图片，其中每个像素点值只能为0,1,2,3且是整数类型，不能超过上面调色板的len()
    :param file_name:str，要存储的话，存储的文件路径
    :param palette:numpy，调色板
    :return:Iamge，经过调色板调色的图片
    '''
    img_using_palette = Image.fromarray(palette[prediction.squeeze().numpy()])
    if file_name != '':
        img_using_palette.save(file_name)
    return img_using_palette


def generate_color_img(ground_truth, prediction, file_name='', palette=VESSEL_PALETTE):
    '''
    :param ground_truth:tensor[1,W,H]，输入的groundtruth，每个像素的值为0或1且为整数，
                    建议使用transforms.ToTensor()(Image.open(picture_name).convert('1')).to(torch.int64)读入
    :param prediction:tensor[1,W,H]，网络的输出，每个像素的值为0或1且为整数
    :param file_name:str，要存储的话，存储的文件路径
    :param palette:numpy，调色板
    :return:Iamge，经过调色板调色的图片
    '''
    output_tensor = ground_truth + 2 * prediction
    output_img = Image.fromarray(palette[output_tensor.squeeze().numpy()])
    # 存储
    if file_name != '':
        output_img.save(file_name)
    return output_img


if __name__ == '__main__':
    # 找个图片测试一下
    mask_dir = 'E:/Datasets/DRIVE/test/2nd_manual/01_manual2.gif'
    mask = transforms.ToTensor()(Image.open(mask_dir).convert('1')).to(torch.int64)
    val_range(mask, 'test')
    paletted_img = color_img(mask)
    img_show(paletted_img)
