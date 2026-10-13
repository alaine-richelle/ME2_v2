"""A compact depthwise-separable CNN classifier."""

from __future__ import annotations

from torch import Tensor, nn


class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(inputs))))


class TinyAudioCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU6(inplace=True),
            SeparableBlock(width, width * 2, stride=2),
            SeparableBlock(width * 2, width * 2, stride=1),
            SeparableBlock(width * 2, width * 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width * 3 * 4, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.dropout(self.features(inputs).flatten(1)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
