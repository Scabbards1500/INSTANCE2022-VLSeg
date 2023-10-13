import torch
import torch.nn as nn
import torch.nn.functional as F


class CovNEXT(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CovNEXT, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(dim=2)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeformableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1):
        super(DeformableConv3D, self).__init__()
        self.conv_offset = nn.Conv3d(in_channels, deformable_groups * 3 * kernel_size ** 3, kernel_size, stride,
                                     padding, dilation)
        self.conv_mask = nn.Conv3d(in_channels, deformable_groups * kernel_size ** 3, kernel_size, stride, padding,
                                   dilation)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.kernel_size = _pair(kernel_size)
        self.deformable_groups = deformable_groups

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.conv_mask(x)
        x = self.conv(x, offset, mask)
        return x


