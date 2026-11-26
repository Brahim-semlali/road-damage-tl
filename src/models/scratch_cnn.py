"""Baseline CNN trained from scratch (no pretrained weights).

Architecture: 5 convolutional blocks [Conv3x3-BN-ReLU] x2 + MaxPool, channels 32→512,
then global average pooling, dropout and a linear classifier (~4.7 M parameters).
It is deliberately simple: it measures what can be learned from our ~1.4k training images
alone, which is the reference point for the benefit of transfer learning.
"""
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int, channels=(32, 64, 128, 256, 512), dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 3
        for c_out in channels:
            layers.append(conv_block(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*layers)          # 224 -> 7 after 5 poolings
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

        for m in self.modules():  # He initialisation for ReLU networks
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
