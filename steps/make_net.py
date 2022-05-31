import time

import torch

from models.unet import create_unet_model
from models.r2unet import *
from utils.timer import Timer


class MakeNet:
    def __init__(self, args, train_loader_lenth):
        """
        :param train_loader_lenth: int 主要是学习率更新策略要用到训练数据的长度
        """
        # 打印开始消息并开始计时
        print("Start making {} network...".format(args.back_bone))
        self.timer = Timer('Stage: Make Network ')

        self.model = None
        # 根据输入数据判断执行哪个backbone
        if args.back_bone == "fcn":
            self.make_fcn(args)
        elif args.back_bone == "unet":
            self.make_unet(args)
        elif args.back_bone == "r2unet":
            self.make_r2unet(args)
        elif args.back_bone == "attunet":
            self.make_attunet(args)
        elif args.back_bone == "r2attunet":
            self.make_r2attunet(args)

        # 追踪迭代的变量作为一个list
        self.params_to_optimize = []
        self.params_trace()
        # optim.SGD需要提供params, lr, momentum, dampening, weight_decay, nesterov
        self.optimizer = torch.optim.SGD(
            self.params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None

        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        # 这里的step数规定为数据集被切成了几个batch，DRIVE数据集20张图，如果batchsize是4,那么step就是5
        self.lr_scheduler = self.create_lr_scheduler(self.optimizer, train_loader_lenth, args.epochs, warmup=True)

        # 如果是继续训练，则读取之前的参数
        if args.resume != "":
            self.resume_data_load(args)

        # 如果要加载预训练模型
        if args.pretrained != "":
            pth = torch.load(args.pretrained, map_location=args.device)
            self.model.load_state_dict(pth['model'])

        # 打印完成消息
        print("Making net finished at: " + str(time.ctime(self.timer.get_current_time())))
        print("Time using: " + str(self.timer.get_stage_elapsed()))
        print('Done.')

    def make_fcn(self, args):
        pass

    def make_unet(self, args):
        self.model = create_unet_model(num_classes=args.num_classes)
        # self.model = U_Net(output_ch=args.num_classes)
        self.model.to(args.device)
        print(self.model)

    def make_r2unet(self, args):
        self.model = R2U_Net(output_ch=args.num_classes)
        self.model.to(args.device)
        print(self.model)

    def make_attunet(self, args):
        self.model = AttU_Net(output_ch=args.num_classes)
        self.model.to(args.device)
        print(self.model)

    def make_r2attunet(self, args):
        self.model = R2AttU_Net(output_ch=args.num_classes)
        self.model.to(args.device)
        print(self.model)

    def params_trace(self):
        # 返回哪些值需要梯度更新
        self.params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

    def resume_data_load(self, args):
        # 如果是继续训练，则从这里恢复数据
        checkpoint = torch.load(args.resume, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            self.scaler.load_state_dict(checkpoint["scaler"])

    @staticmethod
    def create_lr_scheduler(optimizer,
                            num_step: int,
                            epochs: int,
                            warmup=True,
                            warmup_epochs=1,
                            warmup_factor=1e-3):
        assert num_step > 0 and epochs > 0
        if warmup is False:
            warmup_epochs = 0

        def f(x):
            """
            根据step数返回一个学习率倍率因子，
            注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
            """
            if warmup is True and x <= (warmup_epochs * num_step):
                alpha = float(x) / (warmup_epochs * num_step)
                # warmup过程中lr倍率因子从warmup_factor -> 1
                return warmup_factor * (1 - alpha) + alpha
            else:
                # warmup后lr倍率因子从1 -> 0
                # 参考deeplab_v2: Learning rate policy
                return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
