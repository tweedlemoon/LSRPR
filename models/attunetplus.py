import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


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


class Pad(nn.Module):
    def __init__(self):
        super(Pad, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return x1


class PadAndCat(nn.Module):
    def __init__(self):
        super(PadAndCat, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


class AttU_Net_Plus(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, base_size: int = 64, sa: bool = False, sa_kernel_size: int = 7):
        super(AttU_Net_Plus, self).__init__()
        self.sa = sa

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv_D(ch_in=img_ch, ch_out=base_size)
        self.Conv2 = DoubleConv_D(ch_in=base_size, ch_out=2 * base_size)
        self.Conv3 = DoubleConv_D(ch_in=2 * base_size, ch_out=4 * base_size)
        self.Conv4 = DoubleConv_D(ch_in=4 * base_size, ch_out=8 * base_size)
        if self.sa:
            self.Conv5 = SingleConv_D(ch_in=8 * base_size, ch_out=16 * base_size)
            self.SpAtt = SpatialAttention(kernel_size=sa_kernel_size)
            self.Conv6 = SingleConv_D(ch_in=16 * base_size, ch_out=16 * base_size)
        else:
            self.Conv5 = DoubleConv_D(ch_in=8 * base_size, ch_out=16 * base_size)

        self.Up5 = UpConv(ch_in=16 * base_size, ch_out=8 * base_size)
        self.Att5 = AttentionBlock(F_g=8 * base_size, F_l=8 * base_size, F_int=4 * base_size)
        self.Up_conv5 = DoubleConv_D(ch_in=16 * base_size, ch_out=8 * base_size)

        self.Up4 = UpConv(ch_in=8 * base_size, ch_out=4 * base_size)
        self.Att4 = AttentionBlock(F_g=4 * base_size, F_l=4 * base_size, F_int=2 * base_size)
        self.Up_conv4 = DoubleConv_D(ch_in=8 * base_size, ch_out=4 * base_size)

        self.Up3 = UpConv(ch_in=4 * base_size, ch_out=2 * base_size)
        self.Att3 = AttentionBlock(F_g=2 * base_size, F_l=2 * base_size, F_int=base_size)
        self.Up_conv3 = DoubleConv_D(ch_in=4 * base_size, ch_out=2 * base_size)

        self.Up2 = UpConv(ch_in=2 * base_size, ch_out=base_size)
        self.Att2 = AttentionBlock(F_g=base_size, F_l=base_size, F_int=base_size // 2)
        self.Up_conv2 = DoubleConv_D(ch_in=2 * base_size, ch_out=base_size)

        self.Conv_1x1 = nn.Conv2d(base_size, output_ch, kernel_size=1, stride=1, padding=0)

        self.PadAndCat = PadAndCat()
        self.Pad = Pad()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        if self.sa:
            x5 = self.Conv5(x5)
            x5 = self.SpAtt(x5)
            x5 = self.Conv6(x5)
        else:
            x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = self.Pad(d5, x4)
        x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        d5 = self.PadAndCat(d5, x4)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.Pad(d4, x3)
        x3 = self.Att4(g=d4, x=x3)
        # d4 = torch.cat((x3, d4), dim=1)
        d4 = self.PadAndCat(d4, x3)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Pad(d3, x2)
        x2 = self.Att3(g=d3, x=x2)
        # d3 = torch.cat((x2, d3), dim=1)
        d3 = self.PadAndCat(d3, x2)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Pad(d2, x1)
        x1 = self.Att2(g=d2, x=x1)
        # d2 = torch.cat((x1, d2), dim=1)
        d2 = self.PadAndCat(d2, x1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # return d1
        return {"out": d1}


def test():
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 3, 584, 565).to(device)

    model = AttU_Net_Plus(output_ch=2, base_size=64, sa=True).to(device)

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
