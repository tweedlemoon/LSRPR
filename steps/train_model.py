from steps.make_data import MakeData
from steps.make_net import MakeNet
from backbones.unet import UNet
import torch
import argparse


def train_model(args, Data: MakeData, Net: MakeNet):
    for epoch in range(args.start_epoch, args.epochs):
        # 因为如果是继续训练，则从start_epoch开始训练，最终到epochs结束

        pass
    pass


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_frequency, scaler=None):
    model.train()

    pass
