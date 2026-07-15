from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small custom CNN used as a local tiny image-classification baseline."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = torch.flatten(features, 1)
        return self.classifier(features)


def tiny_cnn(num_classes: int = 1000) -> TinyCNN:
    return TinyCNN(num_classes=num_classes)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.activation(images + self.block(images))


class ResNet8(nn.Module):
    """Compact custom residual CNN used as a tiny image-classification baseline."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.features(features)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        return self.fc(features)


def resnet8(num_classes: int = 1000) -> ResNet8:
    return ResNet8(num_classes=num_classes)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.block(images)


class DSCNNSmall(nn.Module):
    """Compact depthwise-separable CNN for tiny image classification."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = torch.flatten(features, 1)
        return self.classifier(features)


def ds_cnn_small(num_classes: int = 1000) -> DSCNNSmall:
    return DSCNNSmall(num_classes=num_classes)
