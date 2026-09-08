"""
U-Net architecture for seismic first break picking.
"""

import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    """
    U-Net architecture with explicit padding for shape alignment.
    U-Net convolutional neural network for seismic first-break picking
    formulated as a multi-class semantic segmentation problem.

    This model follows the encoder-decoder architecture of the U-Net
    framework. The encoder progressively extracts hierarchical spatial
    features from the input seismic data while reducing its spatial
    resolution. The decoder then progressively reconstructs the spatial
    resolution and combines high-level semantic features with
    high-resolution features from the corresponding encoder stages through
    skip connections.

    The network is designed for seismic trace data where the input is
    represented as a single-channel 2D image-like tensor. Each spatial
    location is classified into one of ``out_channels`` classes, allowing
    the model to produce a pixel/trace-wise segmentation map for seismic
    first-break identification.

    Architecture:
        The network consists of four encoder stages, a bottleneck, and four
        decoder stages.

        Encoder:
            - ``enc1``: 1 -> 32 feature channels
            - ``enc2``: 32 -> 64 feature channels
            - ``enc3``: 64 -> 128 feature channels
            - ``enc4``: 128 -> 256 feature channels

        Each encoder stage consists of two convolutional layers, with
        Batch Normalization and ReLU activation after each convolution.
        A 2x2 max-pooling operation is applied after each encoder stage
        to reduce the spatial resolution by a factor of two.

        Bottleneck:
            - ``bottleneck``: 256 -> 512 feature channels

        The bottleneck represents the deepest feature representation of
        the input and operates at the lowest spatial resolution.

        Decoder:
            - ``up4`` / ``dec4``: 512 -> 256 channels
            - ``up3`` / ``dec3``: 256 -> 128 channels
            - ``up2`` / ``dec2``: 128 -> 64 channels
            - ``up1`` / ``dec1``: 64 -> 32 channels

        Each decoder stage first performs learned upsampling using a
        transposed convolution. The upsampled representation is then
        concatenated with the corresponding encoder feature map through a
        U-Net skip connection. The concatenated features are processed by
        a convolutional block.

        Output:
            A final 1x1 convolution maps the 32 decoder feature channels
            to ``out_channels`` output classes.

    Input Shape:
        ``(B, C, H, W)`` where:

            - ``B`` is the batch size.
            - ``C`` is the number of input channels, typically ``1`` for
              single-channel seismic data.
            - ``H`` is the seismic sample/time dimension.
            - ``W`` is the trace or spatial dimension.

        The default expected input shape is approximately
        ``(B, 1, 1578, 751)``.

        Because the U-Net contains four downsampling stages, the spatial
        dimensions must be divisible by ``2 ** 4 = 16`` for exact
        encoder-decoder alignment. If the input dimensions are not
        divisible by 16, the model automatically pads the input on the
        bottom and right sides to the nearest dimensions divisible by 16.

    Output Shape:
        ``(B, out_channels, H, W)``

        The output spatial dimensions are restored to exactly match the
        original input dimensions after the decoder. Any padding introduced
        before the forward pass is removed by cropping the output back to
        the original ``H`` and ``W``.

        With the default configuration, the output shape is:

            ``(B, 3, 1578, 751)``

        The output contains raw logits for each segmentation class. No
        softmax activation is applied inside the model. During training,
        these logits can be passed directly to loss functions such as
        ``nn.CrossEntropyLoss``. During inference, ``torch.softmax`` or
        ``torch.argmax`` can be applied to obtain class probabilities or
        predicted class labels.

    Padding Strategy:
        Spatial dimensions are dynamically padded inside ``forward`` when
        necessary. The target dimensions are calculated as the smallest
        dimensions greater than or equal to the input dimensions that are
        divisible by 16.

        Padding is applied only to the bottom and right sides of the input,
        preserving the original seismic data without modification.
        After the decoder produces the segmentation output, the additional
        padded region is cropped so that the final prediction has exactly
        the same spatial dimensions as the original input.

        This approach allows the model to accept inputs with arbitrary
        spatial dimensions while maintaining valid tensor shapes across
        all encoder and decoder levels.

    Skip Connections:
        Skip connections directly transfer high-resolution feature maps
        from the encoder to the corresponding decoder stage. These
        connections preserve fine-grained spatial information that may be
        lost during pooling and are particularly important for precise
        localization of seismic first breaks.

    Memory Layout:
        The final output is explicitly converted to a contiguous tensor
        using ``Tensor.contiguous()``. This is important for compatibility
        with Apple's Metal Performance Shaders (MPS) backend, where certain
        operations may require tensors to have a contiguous memory layout.

    Args:
        in_channels (int, optional):
            Number of channels in the input seismic data. Defaults to ``1``.

        out_channels (int, optional):
            Number of segmentation classes produced by the network.
            Defaults to ``3``.

    Attributes:
        enc1 (nn.Sequential):
            First encoder convolutional block.

        enc2 (nn.Sequential):
            Second encoder convolutional block.

        enc3 (nn.Sequential):
            Third encoder convolutional block.

        enc4 (nn.Sequential):
            Fourth encoder convolutional block.

        pool1 (nn.MaxPool2d):
            First spatial downsampling layer.

        pool2 (nn.MaxPool2d):
            Second spatial downsampling layer.

        pool3 (nn.MaxPool2d):
            Third spatial downsampling layer.

        pool4 (nn.MaxPool2d):
            Fourth spatial downsampling layer.

        bottleneck (nn.Sequential):
            Deepest convolutional feature extraction block.

        up4 (nn.ConvTranspose2d):
            First decoder upsampling layer.

        up3 (nn.ConvTranspose2d):
            Second decoder upsampling layer.

        up2 (nn.ConvTranspose2d):
            Third decoder upsampling layer.

        up1 (nn.ConvTranspose2d):
            Final decoder upsampling layer.

        dec4 (nn.Sequential):
            First decoder convolutional block.

        dec3 (nn.Sequential):
            Second decoder convolutional block.

        dec2 (nn.Sequential):
            Third decoder convolutional block.

        dec1 (nn.Sequential):
            Final decoder convolutional block.

        out_conv (nn.Conv2d):
            1x1 convolution that projects decoder features to the desired
            number of segmentation classes.

    Example:
        >>> model = UNet(in_channels=1, out_channels=3)
        >>> x = torch.randn(4, 1, 1578, 751)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 3, 1578, 751])

        For multi-class segmentation with ``CrossEntropyLoss``:

        >>> criterion = nn.CrossEntropyLoss()
        >>> targets = torch.randint(0, 3, (4, 1578, 751))
        >>> loss = criterion(logits, targets)

        During inference, class predictions can be obtained with:

        >>> predictions = torch.argmax(logits, dim=1)
        >>> predictions.shape
        torch.Size([4, 1578, 751])
    
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # Encoder
        self.enc1 = self._block(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = self._block(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = self._block(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = self._block(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(256, 512, kernel_size=3, padding=1)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._block(512, 256, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 32, kernel_size=3, padding=1)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels, kernel_size, padding):
        """Convolutional block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # Pad to ensure divisibility by 16
        target_h = ((h + 15) // 16) * 16
        target_w = ((w + 15) // 16) * 16
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        # Output
        out = self.out_conv(d1)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        # ⚠️ CRITICAL: Make contiguous for MPS
        return out.contiguous()
