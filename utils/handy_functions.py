import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def cv2_img_show(img):
    """
    使用opencv展示图片（不推荐）
    :param img:图片
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img)
    cv2.waitKey(0)


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


if __name__ == '__main__':
    pic = '../cache/testimg/deeplab1.png'
    img_PIL = PIL.Image.open(pic)
    img_show(format_convert(img_PIL))
    # img_show(img_PIL)

    img_PIL = img_PIL.convert('RGB')

    img_opencv = cv2.imread(pic)
    img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)

    img_plt = plt.imread(pic)

    img_tensor = transforms.ToTensor()(img_PIL)
    img_numpy = np.array(img_PIL)
    val_range('tensor img', img_numpy)
    pass
