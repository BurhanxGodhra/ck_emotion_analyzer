"""
EEGNet architecture — ported directly from mi-bci-pipeline
(https://github.com/BurhanxGodhra/mi-bci-pipeline/blob/main/src/training/eegnet.py),
unchanged. It's already dataset-agnostic (parameterized by n_classes,
n_channels, n_timepoints), so no modification was needed to reuse it for
DREAMER's 14-channel EEG instead of the original 22-channel motor-imagery data.

Reference: Lawhern et al. (2018), "EEGNet: A Compact Convolutional Network
for EEG-based Brain-Computer Interfaces", Journal of Neural Engineering.
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 14,       # DREAMER's Emotiv EPOC has 14 EEG channels (was 22 for BCI IV-2a)
        n_timepoints: int = 512,    # 4s window at 128Hz (was 1001 for the motor-imagery source project)
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            feat_dim = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        x = self._forward_features(x)
        return self.classifier(x)
