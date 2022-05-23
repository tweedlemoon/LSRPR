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
from backbones.r2unet import *
import PIL
from torchvision import transforms

from utils.eval_utils import ConfusionMatrix
from utils.timer import Timer
from steps.make_data import MakeData

# img_path = Data_Path + "DRIVE/test/images/02_test.tif"
# roi_mask_path = Data_Path + "DRIVE/test/mask/02_test_mask.gif"
# ground_truth_path = Data_Path + "DRIVE/test/1st_manual/02_manual1.gif"

Model_path = 'experimental_data/attunet/1e-6levelset/1/model-attunet-coe-1e-06-time-20220523-165350-best_dice-0.8230832815170288.pth'
Model = 'attunet'
Data_Name = 'DRIVE'


# Model_path = 'save_weights/best_model_author.pth'


def parse_arguments():
    parser = argparse.ArgumentParser(description="inference your model.")
    # batchsize是做数据的时候判断使用多少CPU核心时用的，其实在做验证集时并不需要
    parser.add_argument("-b", "--batch-size", default=Batch_Size, type=int)
    parser.add_argument("--which-gpu", default=Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--data-path", default=Data_Root, help="data root")
    parser.add_argument("--device", default=str(Device), type=str, help="training device")
    # parser.add_argument('--model_path', default="save_weights/best_model_mine.pth", help="the best trained model root")
    # parser.add_argument('--model_path', default="save_weights/best_model_author.pth", help="the best trained model root")
    # parser.add_argument('--model_path', default="save_weights/best_model.pth", help="the best trained model root")
    parser.add_argument('--model_path', default=Model_path, help="the best trained model root")
    parser.add_argument("--back-bone", default=Model, type=str,
                        choices=["fcn", "unet", "r2unet", "attunet", "r2attunet"])
    parser.add_argument("--num-classes", default=Class_Num, type=int)
    parser.add_argument("--dataset", default=Data_Name, type=str, choices=["voc2012", "DRIVE"],
                        help="which dataset to use")

    return parser.parse_args()


def create_model(args):
    if args.back_bone == 'unet':
        return create_unet_model(num_classes=args.num_classes + 1)
    elif args.back_bone == 'r2unet':
        return R2U_Net(output_ch=args.num_classes + 1)
    elif args.back_bone == 'attunet':
        return AttU_Net(output_ch=args.num_classes + 1)
    elif args.back_bone == 'r2attunet':
        return R2AttU_Net(output_ch=args.num_classes + 1)


def compute_index(args):
    matrix = ConfusionMatrix(num_classes=args.num_classes + 1)

    device = args.device
    model = create_model(args=args).to(device)

    pth = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pth['model'])

    # 多张图片测试，直接制作dataloader
    val_loader = MakeData(args=args).val_loader

    all_f1_score = 0.0
    all_accuracy = 0.0
    print('--------------------------------')
    timer = Timer('Evaluating...')
    model.eval()
    with torch.no_grad():
        for idx, (img, real_result) in enumerate(val_loader, start=0):
            ground_truth = val_loader.dataset.manual[idx]
            ground_truth = transforms.ToTensor()(PIL.Image.open(ground_truth).convert('1')).to(torch.int64)
            ground_truth = ground_truth.to(device)

            img = img.to(device)
            net_output = model(img)['out']
            # 进行argmax操作
            argmax_output = net_output.argmax(1)

            matrix.update(ground_truth, argmax_output)
            matrix.prf_compute()
            all_accuracy += matrix.accuracy
            all_f1_score += matrix.f1_score
            matrix.reset()

    accuracy = all_accuracy / val_loader.__len__()
    f1_score = all_f1_score / val_loader.__len__()
    print("Time used: " + str(timer.get_stage_elapsed()))

    print('Report:')
    print('Accuracy:', accuracy.item())
    print('F1 Score:', f1_score.item())
    print('--------------------------------')


def run_inference(args):
    device = args.device
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # load model
    # model = create_unet_model(num_classes=args.num_classes + 1).to(device)
    model = create_model(args=args).to(device)

    pth = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pth['model'])

    # 多张图片测试，直接制作dataloader
    val_loader = MakeData(args=args).val_loader
    model.eval()
    with torch.no_grad():
        for idx, (img, real_result) in enumerate(val_loader, start=0):
            original_img = val_loader.dataset.img_list[idx]
            ground_truth = val_loader.dataset.manual[idx]
            roi_mask = val_loader.dataset.roi_mask[idx]
            original_img = PIL.Image.open(original_img)
            ground_truth = PIL.Image.open(ground_truth)
            # double_img_show(format_convert(original_img), format_convert(ground_truth))

            img = img.to(device)
            net_output = model(img)['out']
            val_range("Network output", net_output)
            # 进行argmax操作
            argmax_output = net_output.argmax(1)
            # val_range("Argmax output", argmax_output)
            # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
            np_argmax_output = np.array(argmax_output.cpu())
            # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
            np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
            # val_range("Numpy argmax output", np_argmax_output)

            # 把周围跟mask处理一下
            roi_img = PIL.Image.open(roi_mask).convert('L')
            roi_img = np.array(roi_img)
            np_argmax_output[roi_img == 0] = 0

            triple_img_show(original_img=format_convert(original_img),
                            original_mask=format_convert(ground_truth),
                            predicted_img=format_convert(np_argmax_output))

    # # 单张图片测试
    # # load picture
    # img_input = PIL.Image.open(img_path).convert('RGB')
    # val_range(name="input image", input_val=img_input)
    # data_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize(mean=mean, std=std)])
    # img = data_transform(img_input)
    # # batch_size=1扩充出来
    # img = torch.unsqueeze(img, dim=0).to(device)
    # val_range("standard img", img)
    # model.eval()
    # with torch.no_grad():
    #     # 送入网络后输出，把0维度弄掉，因为batch_size是1
    #     net_output = model(img)['out']
    #     val_range("Network output", net_output)
    #
    #     # 进行argmax操作
    #     argmax_output = net_output.argmax(1)
    #     val_range("Argmax output", argmax_output)
    #     # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
    #     np_argmax_output = np.array(argmax_output.cpu())
    #     # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
    #     np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
    #     val_range("Numpy argmax output", np_argmax_output)
    #
    #     # 把周围跟mask处理一下
    #     roi_img = PIL.Image.open(roi_mask_path).convert('L')
    #     roi_img = np.array(roi_img)
    #     np_argmax_output[roi_img == 0] = 0
    #
    #     # 载入原图和标准gt，三者展示对比
    #     original_img = PIL.Image.open(img_path)
    #     ground_truth = PIL.Image.open(ground_truth_path)
    #     triple_img_show(format_convert(original_img), format_convert(ground_truth), format_convert(np_argmax_output))


if __name__ == '__main__':
    args = parse_arguments()
    # 当显存不够时使用
    # args.device = 'cpu'
    compute_index(args=args)
    run_inference(args=args)
