import datetime
import os

import torch

from src.loss import criterion
from steps.make_data import MakeData
from steps.make_net import MakeNet
from utils.eval_utils import ConfusionMatrix, DiceCoefficient
from utils.metric_logger import MetricLogger, SmoothedValue
from utils.timer import Timer


def train_eval_model(args, Data: MakeData, Net: MakeNet):
    best_dice = 0.
    timer = Timer("Start training...")
    # results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = "train-result-model-{}-coe-{}-time-{}.txt" \
        .format(args.back_bone,
                args.level_set_coe,
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
    for epoch in range(args.start_epoch, args.epochs):
        # 因为如果是继续训练，则从start_epoch开始训练，最终到epochs结束
        # 训练一个epoch
        # model是要反向传播的，所以必须带着Net传进去，optimizer和lr是实时更新的，所以也要带着类传进去
        mean_loss, lr = train_one_epoch(args=args, model=Net.model, optimizer=Net.optimizer,
                                        data_loader=Data.train_loader,
                                        device=args.device, epoch=epoch, num_classes=args.num_classes,
                                        lr_scheduler=Net.lr_scheduler, print_frequency=args.print_freq,
                                        scaler=Net.scaler)

        # 评估一下这个epoch训练得怎么样
        confmat, dice = evaluate(model=Net.model, data_loader=Data.val_loader,
                                 device=args.device, num_classes=args.num_classes)

        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(os.path.join(args.result_root, results_file), "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        # 在存储文件中多加了一些属性，这是为了checkpoint特制的，如果程序中断，可以通过输入参数中的resume来恢复训练
        # 故验证读取pth文件时，只需要读取model一个部分
        # model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)
        # model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        save_file = {"model": Net.model.state_dict(),
                     "optimizer": Net.optimizer.state_dict(),
                     "lr_scheduler": Net.lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = Net.scaler.state_dict()

        if args.save_best is True:
            # torch.save(save_file, "save_weights/best_model.pth")
            torch.save(save_file, "save_weights/model-{}-coe-{}-time{}-best_dice-{}.pth" \
                       .format(args.back_bone,
                               args.level_set_coe,
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                               best_dice,
                               )
                       )
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = timer.get_stage_elapsed()
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_frequency, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
        loss_weight = torch.as_tensor(args.loss_weight, device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_frequency, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # 老版本loss
            # loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            losses = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # the coefficient of level_set_loss
            level_set_coe = args.level_set_coe
            loss = losses["ce_loss"] + losses["dice_loss"] + level_set_coe * losses["level_set_loss"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(ce_loss=losses["ce_loss"].item(),
                             dice_loss=losses["dice_loss"].item(),
                             level_set_loss=losses["level_set_loss"].item(),
                             loss=loss.item(),
                             lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    dice = DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()
