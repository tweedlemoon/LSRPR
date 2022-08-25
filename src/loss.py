import torch
from torch import nn
from torch.nn import functional

from utils.eval_utils import multiclass_dice_coeff, dice_coeff, build_target


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
#     losses = {}
#     for name, x in inputs.items():
#         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#         # loss = functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
#         target = torch.where(target == 255, 0, target)
#         loss = functional.cross_entropy(x, target, weight=loss_weight)
#         if dice is True:
#             dice_target = build_target(target, num_classes, ignore_index)
#             loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
#         losses[name] = loss
#     losses["levelset"] = level_set_loss_compute(inputs)
#     # losses["levelset"] = level_set_loss_compute(inputs, target)
#     if len(losses) == 1:
#         return losses['out']
#
#     return losses["out"] + 1e-6 * losses["levelset"]
#     # return losses['out'] + 0.5 * losses['aux']

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    x = inputs["out"]
    target = torch.where(target == 255, 0, target)
    losses["ce_loss"] = functional.cross_entropy(x, target, weight=loss_weight)
    losses["level_set_loss"] = level_set_loss_compute(inputs)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        losses["dice_loss"] = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return losses


def criterion_supervised(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True,
                         ignore_index: int = -100):
    losses = {}
    x = inputs["out"]
    target = torch.where(target == 255, 0, target)
    losses["ce_loss"] = functional.cross_entropy(x, target, weight=loss_weight)
    losses["level_set_loss"] = level_set_loss_compute_supervised(inputs, target)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        losses["dice_loss"] = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return losses


# class LevelSetLoss(nn.Module):
#     def __init__(self):
#         super(LevelSetLoss, self).__init__()
#
#     def forward(self, measurement, softmax_output):
#         # input size = batch x channel x height x width
#         outshape = softmax_output.shape
#         tarshape = measurement.shape
#         loss = 0.0
#         for ich in range(tarshape[1]):
#             target_ = torch.unsqueeze(measurement[:, ich], 1)
#             target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
#             pcentroid = torch.sum(target_ * softmax_output, (2, 3)) / torch.sum(softmax_output, (2, 3))
#             pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
#             plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
#             pLoss = plevel * plevel * softmax_output
#             loss += torch.sum(pLoss)
#         return loss


class GradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(GradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        # l2损失
        if (self.penalty == "l2"):
            dH = dH ** 2
            dW = dW ** 2

        loss = torch.sum(dH) + torch.sum(dW)
        return loss


# def level_set_loss_compute(net_output: dict, target: torch.Tensor):
def level_set_loss_compute(net_output: dict):
    """
    返回水平集损失
    :param net_output:  batch * class * height * weight
    :return: tensor, loss value
    """
    # # 对target进行处理
    # target = torch.unsqueeze(target, 1)
    # target_back = 1 - target
    # target = torch.cat([target, target_back], dim=1)

    softmaxed = net_output["out"].softmax(1)
    argmaxed = net_output["out"].argmax(1)
    # 将channel0维也就是背景全部设置为负
    back_ground = -softmaxed[:, 0, :, :]
    fore = softmaxed[:, 1:, :, :]
    fore_ground = torch.sum(fore, dim=(1,), keepdim=True)
    fore_ground = fore_ground.squeeze(1)

    # 两者得到，开始计算水平集损失
    measurement = torch.where(argmaxed == 0, back_ground, fore_ground)
    softmax_output = net_output["out"].softmax(1)

    loss = 0.0

    measurement_multi_channel = torch.unsqueeze(measurement, 1)
    measurement_multi_channel = measurement_multi_channel.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                                 softmax_output.shape[2], softmax_output.shape[3])
    # 这里是自监督，预测的前景和背景靠近自身
    pcentroid = torch.sum(measurement_multi_channel * softmax_output, (2, 3)) / torch.sum(softmax_output, (2, 3))

    # # 这里加了groundtruth，在groundtruth的限定下，前景逼近前景，背景逼近背景
    # pcentroid = torch.sum(measurement_multi_channel * target, (2, 3)) / torch.sum(target, (2, 3))

    pcentroid = pcentroid.view(softmax_output.shape[0], softmax_output.shape[1], 1, 1)
    plevel = measurement_multi_channel - pcentroid.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                          softmax_output.shape[2], softmax_output.shape[3])
    pLoss = plevel * plevel * softmax_output
    loss = loss + torch.sum(pLoss)
    # loss.backward()
    return loss


def level_set_loss_compute_supervised(net_output: dict, target: torch.Tensor):
    """
    返回水平集损失（有监督）
    :param net_output:  batch * class * height * weight
    :return: tensor, loss value
    :param target: ground truth
    :return:
    """

    # # 对target进行处理
    # target = torch.unsqueeze(target, 1)
    # target_back = 1 - target
    # target = torch.cat([target, target_back], dim=1)

    softmaxed = net_output["out"].softmax(1)
    argmaxed = net_output["out"].argmax(1)
    # 将channel0维也就是背景全部设置为负
    back_ground = -softmaxed[:, 0, :, :]
    fore = softmaxed[:, 1:, :, :]
    fore_ground = torch.sum(fore, dim=(1,), keepdim=True)
    fore_ground = fore_ground.squeeze(1)

    # 两者得到，开始计算水平集损失
    measurement = torch.where(argmaxed == 0, back_ground, fore_ground)
    softmax_output = net_output["out"].softmax(1)

    loss = 0.0

    measurement_multi_channel = torch.unsqueeze(measurement, 1)
    measurement_multi_channel = measurement_multi_channel.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                                 softmax_output.shape[2], softmax_output.shape[3])
    # 这里是自监督，预测的前景和背景靠近自身
    # pcentroid = torch.sum(measurement_multi_channel * softmax_output, (2, 3)) / torch.sum(softmax_output, (2, 3))

    # 这里加了groundtruth，在groundtruth的限定下，前景逼近前景，背景逼近背景
    target = target.unsqueeze(dim=1)
    target = torch.cat([1 - target, target], dim=1)
    pcentroid = torch.sum(measurement_multi_channel * target, (2, 3)) / torch.sum(target, (2, 3))

    pcentroid = pcentroid.view(softmax_output.shape[0], softmax_output.shape[1], 1, 1)
    plevel = measurement_multi_channel - pcentroid.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                          softmax_output.shape[2], softmax_output.shape[3])
    pLoss = plevel * plevel * target
    loss = loss + torch.sum(pLoss)
    # loss.backward()
    return loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 仿制输入
    # dummy_net_output = torch.autograd.Variable(torch.sigmoid(torch.randn(2, 2, 8, 16)), requires_grad=True).to(device)
    # # 仿制groundtruth
    # dummy_truth = torch.autograd.Variable(torch.ones_like(dummy_net_output)).to(device)
    # print('Input Size :', dummy_net_output.size())

    '''
    dummy_net_output = torch.tensor(
        [[[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], [[1.2, -1.2], [2.0, -2.0]]],
         [[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], [[1.2, -1.2], [2.0, -2.0]]]],
        requires_grad=True).to(device)    
    '''
    dummy_net_output = torch.tensor(
        [[[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], ],
         [[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], ]],
        requires_grad=True).to(device)
    dummy_truth = torch.tensor(
        [[[1, 0], [0, 1]],
         [[1, 1], [0, 0]]], ).to(device)
    # # 评价标准criteria
    # criteria = LevelSetLoss()
    # loss = criteria(dummy_net_output, dummy_truth)
    dummy_input = {}
    dummy_input["out"] = dummy_net_output
    criteria = level_set_loss_compute(dummy_input)
    # criteria = level_set_loss_compute(dummy_input, dummy_truth)

    # print('Loss Value :', loss)
    print('Loss Value :', criteria)
