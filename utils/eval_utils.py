import torch
import torch.nn as nn
import torch.nn.functional as F


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.miou = None

    def update(self, truth, prediction):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=truth.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (truth >= 0) & (truth < n)
            # 这是一个映射，假如n=3，那么会有预测值和实际值均为{0,1,2}，会有实际0预测012，实际1预测012等9种情况
            # 该算法则是一个函数f，将0-0这个行为map到0上，0-1map到1上，0-2map到2，1-0map到3...以此类推直到2-2map到8
            # 最后统计0-8行为总共出现了几次，这9个数就是混淆矩阵想要的9个数
            inds = n * truth[k].to(torch.int64) + prediction[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()
        if self.accuracy is not None:
            self.accuracy.zero_()
        if self.precision is not None:
            self.precision.zero_()
        if self.recall is not None:
            self.recall.zero_()
        if self.f1_score is not None:
            self.f1_score.zero_()
        if self.miou is not None:
            self.miou.zero_()

    def prf_compute(self):
        h = self.mat.float()
        # 正确率为总预测对的（对角线）除以总像素点数量
        self.accuracy = torch.diag(h).sum() / h.sum()
        # 查准率为主对角线上的值除以该值所在列的和（n类就有n个值）
        self.precision = (torch.diag(h) / h.sum(0))
        # 召回率等于主对角线上的值除以该值所在行的和（n类就有n个值）
        self.recall = (torch.diag(h) / h.sum(1))
        # f1score按公式来（n类就有n个值）
        self.f1_score = torch.div((2 * torch.mul(self.precision, self.recall)), (self.precision + self.recall))
        # miou也按照公式来（这里是直接算最后的miou而不是每一类再除）
        self.miou = (torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))).sum() / self.num_classes

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)


class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        # compute the Dice score, ignoring background
        pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        self.cumulative_dice += multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index)
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


if __name__ == '__main__':
    # test confusion matrix
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_predict = torch.tensor([
        [0, 1, 1],
        [2, 2, 1],
        [1, 0, 1],
    ]).to(device)
    dummy_gt = torch.tensor([
        [0, 2, 2],
        [0, 2, 1],
        [1, 0, 0],
    ]).to(device)
    matrix = ConfusionMatrix(num_classes=3)
    matrix.update(dummy_gt, dummy_predict)
    acc_global, acc, iu = matrix.compute()
    matrix.prf_compute()
