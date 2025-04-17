"""
SimpleCNN 是一个 3 层卷积神经网络结构

    输入是 3 通道彩色图像（如 RGB）
    输出是 2 类（猫 or 狗）

结构为：卷积层 + 激活 + 池化 → 全连接层 → 分类输出
"""

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 第一组卷积模块（提取图像边缘/低阶特征） 输入通道数=3（RGB），输出通道=16，卷积核大小=3x3
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 输入通道数=16，输出通道=32  第二组卷积模块（更深层次特征）
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 输入通道数=32，输出通道=64  第三组卷积模块（语义特征）
            nn.Flatten(),
            nn.Linear(64*16*16, 128), nn.ReLU(),  # 从 16384 维 → 128 维
            nn.Linear(128, 2)  # 输出为 2 维（猫 / 狗）
        )

    def forward(self, x):
        return self.net(x)
