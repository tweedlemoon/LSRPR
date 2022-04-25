import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hyper_parameters import *
from utils.handy_functions import *
from backbones.unet import create_unet_model
import PIL
from PIL import Image
from torchvision import transforms

img_path = Data_Path + "DRIVE/test/images/02_test.tif"
roi_mask_path = Data_Path + "DRIVE/test/mask/02_test_mask.gif"
ground_truth_path = Data_Path + "DRIVE/test/1st_manual/02_manual1.gif"


def parse_arguments():
    parser = argparse.ArgumentParser(description="inference your model.")
    parser.add_argument("--which-gpu", default=Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--data-path", default=Data_Root, help="data root")
    parser.add_argument("--device", default=str(Device), type=str, help="training device")
    # parser.add_argument('--model_path', default="save_weights/best_model_mine.pth", help="the best trained model root")
    parser.add_argument('--model_path', default="save_weights/best_model.pth", help="the best trained model root")
    parser.add_argument("--num-classes", default=Class_Num, type=int)

    return parser.parse_args()


def run_inference(args):
    device = args.device
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # load model
    model = create_unet_model(num_classes=args.num_classes + 1).to(device)
    pth = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pth['model'])

    # load picture
    img_input = Image.open(img_path).convert('RGB')
    val_range(name="input image", input_val=img_input)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(img_input)
    # batch_size=1扩充出来
    img = torch.unsqueeze(img, dim=0).to(device)
    val_range("standard img", img)

    model.eval()
    with torch.no_grad():
        # 送入网络后输出，把0维度弄掉，因为batch_size是1
        net_output = model(img)['out']
        val_range("Network output", net_output)

        # 进行argmax操作
        argmax_output = net_output.argmax(1)
        val_range("Argmax output", argmax_output)
        # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
        np_argmax_output = np.array(argmax_output.cpu())
        # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
        np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
        val_range("Numpy argmax output", np_argmax_output)

        # 把周围跟mask处理一下
        # roi_img = Image.open(roi_mask_path).convert('L')
        # roi_img = np.array(roi_img)
        # np_argmax_output[roi_img == 0] = 0

        # 载入原图和标准gt，三者展示对比
        original_img = Image.open(img_path)
        ground_truth = Image.open(ground_truth_path)
        triple_img_show(original_img, ground_truth, np_argmax_output)


if __name__ == '__main__':
    args = parse_arguments()
    run_inference(args=args)
