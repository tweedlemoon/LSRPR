import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

'''
DropBlock is used in SA-Unet as a regularization method,
which is from Google's work:https://arxiv.org/pdf/1810.12890.pdf
'''


# 在下文的类中，携带_D的意思是使用DropBlock进行正则

class DropBlock(nn.Module):
    """
    See: https://arxiv.org/pdf/1810.12890.pdf
    This code is from https://github.com/FrancescoSaverioZuppichini/DropBlock
    """

    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """Compute gamma, eq (1) in the paper
        Args:
            x (Tensor): Input tensor
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class SpatialAttention(nn.Module):
    """
    This is a part of the following article.
    See:https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
    """

    def __init__(self, in_ch=2, out_ch=1, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            # padding='same' is available only in the pytorch=1.9.0+
            # so if your environment is pytorch 1.9.0+ use the first sentence.
            # nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding='same', stride=1),
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=(self.kernel_size // 2, self.kernel_size // 2),
                stride=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        """
        :param x: input tensor (batch_size,channel,weight,height)
        :return: the same shape (batch_size,channel,weight,height)
        """
        # Average Pooling through channels，但在pytorch中不能这么写
        avg_branch = x.mean(1).unsqueeze(1)
        max_branch = x.max(1).values.unsqueeze(1)
        union = torch.cat([avg_branch, max_branch], dim=1)
        union_out = self.conv(union)
        out = x * union_out
        return out


class PadAndCat(nn.Module):
    def __init__(self):
        super(PadAndCat, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        pad x1 if the shape of x1 is different from x2
        then return the concat of x1 and x2 in the dim 1
        """
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


class DoubleConv_D(nn.Module):
    """
    Double convolution with the DropBlock in it.
    """

    def __init__(self, ch_in, ch_out, block_size=7, keep_prob=0.9):
        super(DoubleConv_D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            DropBlock(block_size=block_size, p=keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            DropBlock(block_size=block_size, p=keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SingleConv_D(nn.Module):
    """
    Single convolution with the DropBlock in it.
    """

    def __init__(self, ch_in, ch_out, block_size=7, keep_prob=0.9):
        super(SingleConv_D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            DropBlock(block_size=block_size, p=keep_prob),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# class UpConv_D(nn.Module):
#     def __init__(self, ch_in, ch_out, block_size=7, keep_prob=0.9):
#         super(UpConv_D, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             DropBlock(block_size=block_size, p=keep_prob),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.up(x)
#         return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class SA_Unet(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, base_size: int = 16, sa=True, sa_kernel_size=7):
        super(SA_Unet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv_D(ch_in=img_ch, ch_out=base_size)
        self.Conv2 = DoubleConv_D(ch_in=base_size, ch_out=2 * base_size)
        self.Conv3 = DoubleConv_D(ch_in=2 * base_size, ch_out=4 * base_size)

        # 最底下是一个single+一个spatial+一个single
        self.Conv4 = SingleConv_D(ch_in=4 * base_size, ch_out=8 * base_size)
        self.SpAtt = SpatialAttention(kernel_size=sa_kernel_size) if sa else SingleConv_D(ch_in=128, ch_out=128)
        self.Conv5 = SingleConv_D(ch_in=8 * base_size, ch_out=8 * base_size)

        self.Up3 = UpConv(ch_in=8 * base_size, ch_out=4 * base_size)
        self.Up_conv3 = DoubleConv_D(ch_in=8 * base_size, ch_out=4 * base_size)
        self.Up2 = UpConv(ch_in=4 * base_size, ch_out=2 * base_size)
        self.Up_conv2 = DoubleConv_D(ch_in=4 * base_size, ch_out=2 * base_size)
        self.Up1 = UpConv(ch_in=2 * base_size, ch_out=base_size)
        self.Up_conv1 = DoubleConv_D(ch_in=2 * base_size, ch_out=base_size)

        self.Conv_1x1 = nn.Conv2d(base_size, output_ch, kernel_size=1, stride=1, padding=0)

        self.PadAndCat = PadAndCat()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        # 最下面三步
        x4 = self.Conv4(x4)
        x4 = self.SpAtt(x4)
        x4 = self.Conv5(x4)

        # decoding + concat path
        d3 = self.Up3(x4)
        d3 = self.PadAndCat(d3, x3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.PadAndCat(d2, x2)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = self.PadAndCat(d1, x1)
        d1 = self.Up_conv1(d1)

        out = self.Conv_1x1(d1)

        return {"out": out}


def test():
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 3, 584, 565).to(device)

    model = SA_Unet(output_ch=2, base_size=64).to(device)

    output = model(dummy_input)

    print(model)
    print('\nModel input shape :', dummy_input.size())
    print('Model output shape :', output["out"].size())


if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # test_tensor = torch.tensor(
    #     [[[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], ],
    #      [[[1.0, 0.9], [2.0, -1.0]], [[1.3, 0], [1.5, 0.5]], ]],
    #     requires_grad=True).to(device)
    # print(test_tensor)
    # # max_test = test_tensor.max(1).values
    # mean_test = test_tensor.mean(1)

    test()
