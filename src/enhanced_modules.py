import torch
import torch.nn as nn
import timm
from efficientnet_pytorch import EfficientNet
import os
import torch.nn.functional as F
from typing import Tuple

# 空间变换网络 (STN) 定义
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # 这里的输入维度调整为 10 * 52 * 52
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),  # 27040 是 10 * 52 * 52 的计算结果
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        x_size_lst = list(xs.size())
        total_size = x_size_lst[0] * x_size_lst[1] * x_size_lst[2] * x_size_lst[3]
        xs = xs.view(x_size_lst[0], total_size // x_size_lst[0])  # 展开成二维张量
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
