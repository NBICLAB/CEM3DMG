import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class Conv_block3D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block3D, self).__init__()
        self.conv1 = nn.Conv3d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm3d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv3d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm3d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv3d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm3d(n_ch_out, momentum=m)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x


class Up3D(nn.Module):
    def __init__(self, n_ch):
        super(Up3D, self).__init__()
        self.cov_1 = nn.Conv3d(n_ch, n_ch, 3, 1, 0)
        self.ac = nn.LeakyReLU(0.2, True)
        self.bn1 = nn.BatchNorm3d(n_ch)

    def forward(self, x):
        x = self.ac(self.bn1(self.cov_1(F.interpolate(x, scale_factor=2, mode='nearest'))))
        return x



class test_Generator3(nn.Module):
    def __init__(self, in_channel, step_channel=8):
        super(test_Generator3, self).__init__()
        self.in_channel = in_channel

        self.body1_1 = Conv_block3D(in_channel, step_channel)
        self.body1_2 = Conv_block3D(step_channel * 2, step_channel * 2)
        self.body1_3 = Conv_block3D(step_channel * 3, step_channel * 3)
        self.body1_4 = Conv_block3D(step_channel * 4, step_channel * 4)

        self.body2_1 = Conv_block3D(in_channel, step_channel)
        self.body3_1 = Conv_block3D(in_channel, step_channel)
        self.body4_1 = Conv_block3D(in_channel, step_channel)

        self.upbn1 = Up3D(step_channel)
        self.upbn2 = Up3D(2 * step_channel)
        self.upbn3 = Up3D(3 * step_channel)

        self.last_covs = nn.Sequential(
            nn.Conv3d(step_channel * 4, step_channel* 2, 5, 1, 0, bias=True),
            nn.BatchNorm3d(step_channel* 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(step_channel * 2, step_channel, 3, 1, 0, bias=True),
            nn.BatchNorm3d(step_channel),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(step_channel, in_channel, 1, 1, 0, bias=True),
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        out_1 = self.body1_1(x[0])
        out_1 = self.upbn1(out_1)
        out_2_1 = self.body2_1(x[1])

        out_1 = torch.cat([out_1, out_2_1], dim=1)

        out_1 = self.body1_2(out_1)
        out_1 = self.upbn2(out_1)
        out_3_1 = self.body3_1(x[2])
        out_1 = torch.cat([out_1, out_3_1], dim=1)

        out_1 = self.body1_3(out_1)
        out_1 = self.upbn3(out_1)
        out_4_1 = self.body4_1(x[3])
        out_1 = torch.cat([out_1, out_4_1], dim=1)

        out_1 = self.body1_4(out_1)

        out_1 = self.last_covs(out_1)
        out_1 = self.tanh(out_1)

        return out_1


class VD(nn.Module):
    def __init__(self, in_channel, pad):
        super(VD, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)

        self.pool1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)

        self.cov = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, inputs):
        out = self.relu(self.conv1_1(inputs))
        out = self.relu(self.conv1_2(out))
        out = self.relu(self.pool1(out))

        out = self.relu(self.conv2_1(out))
        out = self.relu(self.conv2_2(out))
        out = self.relu(self.pool2(out))

        out = self.relu(self.conv3_1(out))
        out = self.relu(self.conv3_2(out))
        out = self.relu(self.conv3_3(out))
        out = self.relu(self.conv3_4(out))
        out = self.relu(self.pool3(out))

        out = self.cov(out)
        return out

