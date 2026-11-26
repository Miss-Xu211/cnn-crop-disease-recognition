"""
CBAM (Convolutional Block Attention Module) 独立实现
=====================================================
本文件只包含 CBAM 注意力模块，方便单独阅读和理解。
论文: CBAM: Convolutional Block Attention Module (ECCV 2018)

依赖安装:
    pip install torch

CBAM 的核心思想:
    特征图先经过【通道注意力】判断"哪些通道(特征)重要"，
    再经过【空间注意力】判断"图片上哪些位置重要"，
    两次都用"注意力权重 × 原特征"的方式做加权，突出有用信息、抑制无关信息。

    输入 x 的形状: (B, C, H, W)
        B = batch size, C = 通道数, H/W = 高/宽
    输出形状: (B, C, H, W) —— 形状不变，可直接插在任意卷积层后面。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力 (Channel Attention Module)
    -------------------------------------
    做法:
      1. 对每个通道做全局平均池化 -> 得到 (B, C, 1, 1) 的"平均特征"
      2. 对每个通道做全局最大池化 -> 得到 (B, C, 1, 1) 的"最大特征"
      3. 两个结果分别经过同一个共享的两层 MLP(1x1 卷积实现，先降维再升维)
      4. 相加后经 Sigmoid，得到 0~1 的通道权重 (B, C, 1, 1)
      5. 与输入逐通道相乘，完成通道维度的加权
    直觉:
      平均池化反映"整体响应强度"，最大池化反映"最显著的特征"，
      两者互补，能更全面地判断通道的重要性。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 中间层通道数，至少要保留 4 个通道，避免通道太少时降维到 0
        hidden = max(channels // reduction, 4)

        # 两种全局池化，把 H×W 压成 1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP: C -> hidden -> C，用 1x1 卷积等价实现全连接
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = self.shared_mlp(self.avg_pool(x))   # (B, C, 1, 1)
        max_out = self.shared_mlp(self.max_pool(x))   # (B, C, 1, 1)
        weight = self.sigmoid(avg_out + max_out)      # (B, C, 1, 1)，取值 0~1
        return x * weight                             # 广播相乘，按通道加权


class SpatialAttention(nn.Module):
    """
    空间注意力 (Spatial Attention Module)
    -------------------------------------
    做法:
      1. 沿通道维度分别求平均和最大值 -> 两个 (B, 1, H, W) 的二维图
      2. 把两张图拼接成 (B, 2, H, W)
      3. 用 7x7 卷积融合成 (B, 1, H, W)，再 Sigmoid 得到 0~1 的空间权重
      4. 与输入逐位置相乘，完成空间维度的加权
    直觉:
      "哪里有病害/目标"这一信息在通道平均图和最大值图上都能体现，
      卷积可以把这种位置响应转成一张注意力热力图。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        # 输入 2 个通道(平均图 + 最大图)，输出 1 个通道的注意力图
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维(dim=1)压缩
        avg_out = torch.mean(x, dim=1, keepdim=True)          # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # (B, 1, H, W)
        feat = torch.cat([avg_out, max_out], dim=1)           # (B, 2, H, W)
        weight = self.sigmoid(self.conv(feat))                # (B, 1, H, W)
        return x * weight                                     # 广播相乘，按位置加权


class CBAM(nn.Module):
    """
    完整的 CBAM 模块 = 通道注意力 + 空间注意力（串联）
    -----------------------------------------------
    参数:
        channels   : 输入特征图的通道数
        reduction  : 通道注意力中 MLP 的降维比例，默认 16
        kernel_size: 空间注意力的卷积核大小，默认 7
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先用通道注意力加权，再用空间注意力加权（顺序可互换，论文推荐先通道后空间）
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# 直接运行本文件可做一次形状自检，帮助理解模块输入输出
# 运行方式: python cbam.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = torch.randn(2, 64, 32, 32)       # 模拟一批特征图: B=2, C=64, H=W=32
    cbam = CBAM(channels=64)
    out = cbam(demo)

    print("输入形状:", tuple(demo.shape))
    print("输出形状:", tuple(out.shape))     # 与输入完全一致
    print("CBAM 参数量:", sum(p.numel() for p in cbam.parameters()))
