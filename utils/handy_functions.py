import torch
import numpy as np
import cv2
from torchvision import transforms


def val_range(name: str, input_val):
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
    展示图片
    :param img:图片
    """
    if img.type() == "torch.LongTensor":
        img_numpy = np.array(img, dtype=np.uint8)
    elif img.type() == "torch.FloatTensor":
        img_numpy = np.array(img)
    else:
        return -1
    # cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
    cv2.imshow("show img", img_numpy)
    cv2.waitKey(0)


def triple_img_show(original_img, original_mask, pridiction_img):
    """
    :param original_img: 原图
    :param original_mask: 原mask
    :param pridiction_img: 预测结果
    """
    original_img = np.array(original_img)
    original_mask = np.array(original_mask)
    pridiction_img = np.array(pridiction_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)
    pridiction_img = cv2.cvtColor(pridiction_img, cv2.COLOR_BGR2RGB)
    # 展示图片
    cv2.imshow("Original image", original_img)
    cv2.imshow("Original mask", original_mask)
    cv2.imshow("Prediction", pridiction_img)
    cv2.waitKey(0)
