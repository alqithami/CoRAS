from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetEncoder(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False):
        super().__init__()
        if name != "resnet18":
            raise ValueError("Only resnet18 is implemented in this package. Use small_cnn for fast local runs.")
        try:
            import torchvision.models as tvm
        except Exception as exc:
            raise RuntimeError("torchvision is required for encoder=resnet18") from exc
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
        self.out_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FeaturePromptAdapter(nn.Module):
    """Feature-level prompt/adaptor for frozen robot perception backbones.

    It is intentionally small: a learned prompt vector plus a low-rank residual.
    This instantiates the prompt-adaptation stage while avoiding full fine-tuning.
    """

    def __init__(self, dim: int, rank: int = 16, prompt_scale: float = 0.01):
        super().__init__()
        self.prompt = nn.Parameter(prompt_scale * torch.randn(dim))
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.prompt + self.up(torch.relu(self.down(h))))


class ActionTokenModel(nn.Module):
    def __init__(self, num_classes: int, encoder: str = "small_cnn", pretrained: bool = False, adapter_rank: int = 16):
        super().__init__()
        if encoder == "small_cnn":
            self.encoder = SmallCNN(out_dim=256)
        elif encoder == "resnet18":
            self.encoder = ResNetEncoder("resnet18", pretrained=pretrained)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        self.adapter = FeaturePromptAdapter(self.encoder.out_dim, rank=adapter_rank)
        self.head = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.adapter(h)
        return self.head(h)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_head(self) -> None:
        for p in self.head.parameters():
            p.requires_grad = False

    def train_only_adapter(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True

    def train_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)
