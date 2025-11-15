# models/tcn_baseline.py
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """remove extra elements at the end after padding for causal conv"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """
    input:  (b, c, t)
    output: (b, hidden_dim)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        current_in = in_channels

        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_in = hidden_channels

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)       # (b, hidden, t)
        out = self.global_pool(out) # (b, hidden, 1)
        out = out.squeeze(-1)       # (b, hidden)
        return out


class TCNClassifier(nn.Module):
    """
    input:  (b, c, t)
    output: (b, num_classes) logits
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        classification_type: str | None = None,
    ):
        super().__init__()
        self.tcn = TCNBackbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_channels, num_classes)
        self.classification_type = classification_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        logits = self.head(features)
        return logits
