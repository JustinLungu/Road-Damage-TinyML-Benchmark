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
