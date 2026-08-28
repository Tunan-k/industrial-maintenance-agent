"""
功能：
    故障诊断模型
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    """
    1D convolution with manual same-padding.

    Keeps sequence length unchanged when stride=1.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()

        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.stride == 1:

            pad_total = self.kernel_size - 1

            pad_left = pad_total // 2

            pad_right = (
                pad_total - pad_left
            )

            x = F.pad(
                x,
                (pad_left, pad_right),
            )

        return self.conv(x)


class DSBBlock(nn.Module):

    def __init__(
        self,
        in_ch: int,
        n_cls: int,
    ):
        super().__init__()

        self.bn = nn.BatchNorm1d(
            in_ch
        )

        self.gmp = nn.AdaptiveMaxPool1d(
            1
        )

        self.fc = nn.Linear(
            in_ch,
            n_cls,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        x = self.bn(x)

        feat = self.gmp(
            x
        ).squeeze(-1)

        logits = self.fc(
            feat
        )

        return feat, logits


class DSMT1DCNN10(nn.Module):
    """
    DSMT-1DCNN adapted for drilling-pump
    10-class fault diagnosis.

    Input:
        (B, 3, 1024)

    Channel order:
        stress
        vibration
        pressure

    Output:
        final logits: (B, 10)
    """

    def __init__(
        self,
        in_ch: int = 3,
        n_cls: int = 10,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            SamePadConv1d(
                in_ch,
                32,
                kernel_size=128,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            SamePadConv1d(
                32,
                32,
                kernel_size=3,
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            SamePadConv1d(
                32,
                32,
                kernel_size=3,
            ),
            nn.ReLU(),
        )

        self.dsb1 = DSBBlock(
            32,
            n_cls,
        )

        self.dsb2 = DSBBlock(
            32,
            n_cls,
        )

        self.dsb3 = DSBBlock(
            32,
            n_cls,
        )

        self.fusion = nn.Linear(
            32 * 3,
            96,
        )

        self.head = nn.Linear(
            96,
            n_cls,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)

        g1, d1 = self.dsb1(f1)
        g2, d2 = self.dsb2(f2)
        g3, d3 = self.dsb3(f3)

        fusion_feat = self.fusion(
            torch.cat(
                [g1, g2, g3],
                dim=1,
            )
        )

        logits = self.head(
            fusion_feat
        )

        return (
            d1,
            d2,
            d3,
        ), logits