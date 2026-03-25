"""
Attention modules for YOLOv8 small object detection experiments.
Place this file at: ultralytics/nn/modules/attention.py

Three attention mechanisms:
  - SimAM:    Parameter-free 3D attention (zero overhead)
  - CBAM:     Channel + Spatial attention (sequential)
  - CoordAtt: Coordinate Attention with positional encoding
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM) - Parameter-free.
    Computes 3D attention weights based on neuron energy functions.
    Reference: http://proceedings.mlr.press/v139/yang21o.html
    
    No extra parameters, no extra FLOPs beyond element-wise ops.
    """
    def __init__(self, channels=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        # channels arg is accepted but unused (needed for YAML parsing compatibility)

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        # Compute per-neuron energy
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        # Energy-based importance
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e_inv)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequential channel attention ("what") + spatial attention ("where").
    Reference: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.channel_sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.amax(dim=1, keepdim=True)
        sa = self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        return x * sa


class CoordAtt(nn.Module):
    """
    Coordinate Attention.
    Captures long-range dependencies with directional positional info.
    Particularly useful for localizing small objects.
    Reference: https://arxiv.org/abs/2103.02907
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # H x 1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 1 x W
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.size()
        # Encode horizontal and vertical spatial info
        x_h = self.pool_h(x)                          # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)     # (n, c, w, 1)

        # Concatenate along spatial dim, compress, split
        y = torch.cat([x_h, x_w], dim=2)              # (n, c, h+w, 1)
        y = self.act(self.bn1(self.conv1(y)))          # (n, mid, h+w, 1)
        x_h, x_w = torch.split(y, [h, w], dim=2)      # split back

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))          # (n, c, h, 1)
        a_w = torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))  # (n, c, 1, w)

        return x * a_h * a_w