import torch
import torch.nn as nn

""" Squeeze and Excitation block """


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


""" 3x3->3x3 Residual block """


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4


""" Mixpool block: Merging the image features and the mask """


class MixPool(nn.Module):
    def __init__(self, in_c, out_c):
        super(MixPool, self).__init__()

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = (self.fmask(x) > 0.5).type(torch.cuda.FloatTensor)
        m = nn.MaxPool2d((m.shape[2] // x.shape[2], m.shape[3] // x.shape[3]))(m)
        x1 = x * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], axis=1)
        return


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, name=None):
        super(EncoderBlock, self).__init__()

        self.name = name
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, masks):
        x = self.r1(inputs)
        x = self.r2(x)
        p = self.p1(x, masks)
        o = self.pool(p)
        return o, x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, name=None):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c + in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c)

    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        p = self.p1(x, masks)
        return p


class FANet(nn.Module):
    def __init__(self):
        super(FANet, self).__init__()

        self.e1 = EncoderBlock(3, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)
        self.d4 = DecoderBlock(32, 16)

        self.output = nn.Conv2d(16 + 1, 1, kernel_size=1, padding=0)

    def forward(self, x):
        inputs, masks = x[0], x[1]

        p1, s1 = self.e1(inputs, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)

        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)

        d5 = torch.cat([d4, masks], axis=1)
        output = self.output(d5)

        return {"out": output}


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)).cuda()
    m = torch.randn((2, 1, 256, 256)).cuda()
    model = FANet().cuda()
    y = model([x, m])
    print(y.shape)
