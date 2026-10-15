"""
Spectrogram CNN classifier.

The network consumes a single-channel 128x128 normalised log-Mel image and
emits one logit: the un-squashed evidence that the utterance is synthetic.

Why a CNN over a spectrogram at all
-----------------------------------
Neural vocoders (WaveNet, HiFi-GAN, WORLD, and the concatenative and
voice-conversion systems in the ASVspoof attack set) reconstruct a waveform
from an intermediate acoustic representation. That reconstruction leaves
repeatable, local, texture-like traces: an over-smoothed harmonic stack,
missing or artificially regular energy above roughly 4 kHz, phase that is not
consistent between neighbouring frames, and unnaturally steady formant tracks.
Those traces are local patterns in a time-frequency image, which is exactly the
inductive bias a 2-D convolution encodes - translation-invariant local feature
detection with parameter sharing. A fully connected network on raw samples
would have to learn that invariance from scratch from far more data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VoiceDeepfakeCNN(nn.Module):
    """
    Four convolutional blocks followed by a dense head.

    Output is a raw logit, not a probability. Training pairs this with
    BCEWithLogitsLoss, which folds the sigmoid into the loss using the
    log-sum-exp trick and therefore does not overflow when the model becomes
    confident. Call `predict_proba` when a probability in [0, 1] is wanted.

    The parameter layout is unchanged from v1, so a v1 checkpoint loads here
    without conversion: the removed trailing nn.Sigmoid carried no weights.
    """

    def __init__(self, in_channels: int = 1, dropout_rate: float = 0.25):
        super().__init__()

        # (B,1,128,128) -> (B,32,64,64)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        # (B,32,64,64) -> (B,64,32,32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        # (B,64,32,32) -> (B,128,16,16)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate + 0.05),
        )
        # (B,128,16,16) -> (B,256,4,4)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout_rate + 0.1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, 128) -> (B, 1) logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, 128) -> (B,) probability that each window is spoofed."""
        self.eval()
        return torch.sigmoid(self.forward(x)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = VoiceDeepfakeCNN()
    dummy = torch.randn(4, 1, 128, 128)
    out = net(dummy)
    print("input :", tuple(dummy.shape))
    print("logits:", tuple(out.shape))
    print("probs :", tuple(net.predict_proba(dummy).shape))
    print(f"trainable parameters: {count_parameters(net):,}")
