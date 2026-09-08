"""
MobileNet + U-Net Decoder for seismic segmentation.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileUNet(nn.Module):
    """
    MobileNetV2 encoder + U-Net decoder.

    MobileNetV2 encoder with a U-Net-style decoder for semantic segmentation.

This model is designed for semantic segmentation of seismic images. It uses
a pretrained MobileNetV2 as the encoder and a U-Net-style decoder with skip
connections to progressively restore the spatial resolution of the feature
maps.

Architecture:
Input
↓
1-channel → 3-channel stem
↓
MobileNetV2 Encoder
├── enc1: 16 channels
├── enc2: 24 channels
├── enc3: 32 channels
├── enc4: 96 channels
└── enc5: 320 channels
↓
U-Net Decoder
├── dec5: 320 → 64
├── dec4: 160 → 32
├── dec3: 64 → 24
├── dec2: 48 → 16
└── dec1: 32 → 16
↓
Output convolution
↓
Segmentation mask

Input:
Tensor of shape (B, in_channels, H, W).

If H or W is not divisible by 32, the input is automatically padded
with zeros to the nearest dimensions divisible by 32.

Output:
Tensor of shape (B, out_channels, H, W), where H and W correspond to
the original input spatial dimensions.

Args:
in_channels (int):
Number of input channels. Defaults to 1, which is suitable for
single-channel seismic data.

out_channels (int):
    Number of output segmentation classes. Defaults to 3.

pretrained (bool):
    Whether to initialize the MobileNetV2 encoder with ImageNet
    pretrained weights. Defaults to True.

Notes:
- The MobileNetV2 encoder weights are frozen during training.
- A trainable convolutional stem converts single-channel input to
the three channels expected by MobileNetV2.
- Encoder feature maps are used as skip connections in the decoder.
- ConvTranspose2d layers are used to progressively increase spatial
resolution.
- The final output is cropped back to the original input dimensions
after padding.

Example:
>>> model = MobileUNet(
... in_channels=1,
... out_channels=3,
... pretrained=True,
... )
>>> x = torch.randn(2, 1, 512, 512)
>>> y = model(x)
>>> y.shape
torch.Size([2, 3, 512, 512])
    
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 3, pretrained: bool = True
    ):
        super().__init__()

        # Encoder: MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.encoder = mobilenet_v2(weights=weights)
        else:
            self.encoder = mobilenet_v2(weights=None)

        # Freeze encoder weights (optional)
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("🔒 MobileNet encoder frozen")

        # Expand input to 3 channels
        self.stem = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

        # MobileNetV2 features
        # MaxPool2d(2) silindi, çünki features[0:2] onsuz da stride=2 istifadə edir.
        self.enc1 = self.encoder.features[0:2]  # 16 channels
        self.enc2 = self.encoder.features[2:4]  # 24 channels
        self.enc3 = self.encoder.features[4:7]  # 32 channels
        self.enc4 = self.encoder.features[7:14]  # 96 channels
        self.enc5 = self.encoder.features[14:18]  # 320 channels

        # --- Decoder ---
        self.dec5 = self._decoder_block(320, 64)  # 320 → 64
        self.dec4 = self._decoder_block(160, 32)  # 64 + 96 = 160 → 32
        self.dec3 = self._decoder_block(64, 24)  # 32 + 32 = 64 → 24
        self.dec2 = self._decoder_block(48, 16)  # 24 + 24 = 48 → 16
        self.dec1 = self._decoder_block(32, 16)  # 16 + 16 = 32 → 16

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, _, h, w = x.shape

        # Pad to ensure divisibility by 32
        target_h = ((h + 31) // 32) * 32
        target_w = ((w + 31) // 32) * 32
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        x = self.stem(x)

        e1 = self.enc1(x)  # 16 channels
        e2 = self.enc2(e1)  # 24 channels
        e3 = self.enc3(e2)  # 32 channels
        e4 = self.enc4(e3)  # 96 channels
        e5 = self.enc5(e4)  # 320 channels

        # Decoder with skip connections
        d5 = self.dec5(e5)  # 320 → 64
        e4_up = F.interpolate(
            e4, size=d5.shape[2:], mode="bilinear", align_corners=False
        )
        d5 = torch.cat([d5, e4_up], dim=1)  # 64 + 96 = 160

        d4 = self.dec4(d5)  # 160 → 32
        e3_up = F.interpolate(
            e3, size=d4.shape[2:], mode="bilinear", align_corners=False
        )
        d4 = torch.cat([d4, e3_up], dim=1)  # 32 + 32 = 64

        d3 = self.dec3(d4)  # 64 → 24
        e2_up = F.interpolate(
            e2, size=d3.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = torch.cat([d3, e2_up], dim=1)  # 24 + 24 = 48

        d2 = self.dec2(d3)  # 48 → 16
        e1_up = F.interpolate(
            e1, size=d2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = torch.cat([d2, e1_up], dim=1)  # 16 + 16 = 32

        d1 = self.dec1(d2)  # 32 → 16

        out = self.out_conv(d1)

        # Crop back to original input dimensions
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.contiguous()
