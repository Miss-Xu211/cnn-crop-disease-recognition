"""
两个对比模型的定义文件
======================
模型A (BaselineCNN): 基础 CNN，作为基准模型
模型B (CBAMCNN)    : 完全相同的 CNN 结构，但在每个卷积块后嵌入 CBAM 注意力

对照实验的关键: 两个模型的卷积/池化/全连接结构、通道数、参数量级完全一致，
唯一差别就是是否使用 CBAM，这样对比结果才有意义。

依赖安装:
    pip install torch torchvision
"""

import torch
import torch.nn as nn

from cbam import CBAM


class ConvBlock(nn.Module):
    """
    一个基础卷积块: Conv3x3 -> BatchNorm -> ReLU -> (可选) CBAM -> MaxPool
    ----------------------------------------------------------------
    说明:
      - padding=1、kernel=3 保证卷积前后特征图大小不变，方便插入注意力模块
      - BatchNorm 让训练更稳定、收敛更快
      - use_cbam=True 时，在激活之后、池化之前插入 CBAM（对空间分辨率更高的
        特征图做注意力更有效）
    """

    def __init__(self, in_channels, out_channels, use_cbam=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 模型B的关键：注意力模块（模型A为 nn.Identity，即不做任何处理）
        self.attn = CBAM(out_channels) if use_cbam else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))   # 卷积 + 归一化 + 激活
        x = self.attn(x)                       # 注意力加权（模型A跳过）
        x = self.pool(x)                       # 2x2 最大池化，长宽减半
        return x


class BaselineCNN(nn.Module):
    """
    模型A: 基础 CNN（基准模型）
    -------------------------
    结构:
      4 个卷积块 (32 -> 64 -> 128 -> 256 通道)
      -> 全局平均池化 (把 H×W 压成 1×1)
      -> Dropout(防止过拟合)
      -> 全连接层输出 num_classes 个类别分数
    输入: (B, 3, 128, 128)
    """

    def __init__(self, num_classes=5, use_cbam=False):
        super().__init__()
        # 四个卷积块，每个块后空间尺寸减半：128 -> 64 -> 32 -> 16 -> 8
        self.features = nn.Sequential(
            ConvBlock(3, 32, use_cbam=use_cbam),
            ConvBlock(32, 64, use_cbam=use_cbam),
            ConvBlock(64, 128, use_cbam=use_cbam),
            ConvBlock(128, 256, use_cbam=use_cbam),
        )
        # 全局平均池化：每个通道取平均，得到 (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)          # (B, 256, 8, 8)
        x = self.gap(x)               # (B, 256, 1, 1)
        x = torch.flatten(x, 1)       # (B, 256)
        x = self.dropout(x)
        x = self.fc(x)                # (B, num_classes)
        return x


class CBAMCNN(BaselineCNN):
    """
    模型B: 嵌入 CBAM 的改进 CNN
    --------------------------
    直接复用 BaselineCNN 的骨架，只把 use_cbam 设为 True，
    因此两个模型除注意力模块外结构完全相同，构成严格的对照实验。
    """

    def __init__(self, num_classes=5):
        super().__init__(num_classes=num_classes, use_cbam=True)


# ---------------------------------------------------------------------------
# 形状/参数量自检：python models.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)

    model_a = BaselineCNN(num_classes=5)
    model_b = CBAMCNN(num_classes=5)

    print("模型A 输出:", tuple(model_a(x).shape),
          "参数量:", f"{sum(p.numel() for p in model_a.parameters()):,}")
    print("模型B 输出:", tuple(model_b(x).shape),
          "参数量:", f"{sum(p.numel() for p in model_b.parameters()):,}")
