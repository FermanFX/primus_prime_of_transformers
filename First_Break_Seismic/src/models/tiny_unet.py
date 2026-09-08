"""
Tiny U-Net for quick testing.
Parameters: ~50K only!
"""

import torch
import torch.nn.functional as F
from torch import nn


class TinyUNet(nn.Module):
    """
    Extremely lightweight U-Net for quick testing.
    Parameters: ~50K
    Training time: ~5 minute per epoch
    
    Extremely lightweight U-Net architecture designed for rapid testing,
    prototyping, debugging, and experimentation with seismic segmentation
    models.

    This model is a compact variant of the standard U-Net architecture.
    It preserves the core encoder-decoder structure and skip connections
    while using significantly fewer feature channels than a full-sized
    U-Net. The reduced number of channels makes the model computationally
    inexpensive and suitable for quick experiments where training speed
    and low memory consumption are more important than maximum model
    capacity.

    The network can be used for multi-class semantic segmentation of
    single-channel seismic data, including seismic first-break picking.
    Each spatial location in the input is assigned one of
    ``out_channels`` segmentation classes.

    Architecture:
        The model contains three encoder stages, a bottleneck, and three
        decoder stages.

        Encoder:
            - ``enc1``: ``in_channels`` -> 4 feature channels
            - ``enc2``: 4 -> 8 feature channels
            - ``enc3``: 8 -> 16 feature channels

        Each encoder stage consists of two 3x3 convolutional layers with
        Batch Normalization and ReLU activation. A shared 2x2 max-pooling
        layer is used between encoder stages to reduce the spatial
        resolution.

        Bottleneck:
            - ``bottleneck``: 16 -> 32 feature channels

        The bottleneck is the deepest feature extraction stage and operates
        at the lowest spatial resolution of the network.

        Decoder:
            - ``up3`` / ``dec3``: 32 -> 16 feature channels
            - ``up2`` / ``dec2``: 16 -> 8 feature channels
            - ``up1`` / ``dec1``: 8 -> 4 feature channels

        Each decoder stage uses a transposed convolution for learned
        upsampling. The upsampled feature map is concatenated with the
        corresponding encoder feature map through a skip connection.
        The resulting feature representation is then processed by a
        convolutional block.

        Output Head:
            A final 1x1 convolution maps the 4 decoder feature channels
            to ``out_channels`` segmentation classes.

    Model Capacity:
        The network intentionally uses very small channel dimensions
        (4, 8, 16, and 32). This substantially reduces the number of
        trainable parameters and computational cost compared with a
        conventional U-Net.

        The model is therefore particularly useful for:

            - validating data loading and preprocessing pipelines,
            - testing training loops,
            - debugging loss functions,
            - checking tensor shapes,
            - validating segmentation labels,
            - performing quick overfitting experiments,
            - testing hardware or device compatibility,
            - rapid architecture prototyping.

        The exact parameter count depends on ``in_channels`` and
        ``out_channels``.

    Input Shape:
        ``(B, C, H, W)`` where:

            - ``B`` is the batch size.
            - ``C`` is the number of input channels.
            - ``H`` is the input height, such as seismic time/sample
              dimension.
            - ``W`` is the input width, such as seismic trace/spatial
              dimension.

        By default, the model expects a single-channel input:

            ``(B, 1, H, W)``

        The model does not require the input dimensions to be divisible
        by 8. Instead, the ``forward`` method dynamically pads the input
        when necessary.

    Padding Strategy:
        Because the model performs three consecutive downsampling
        operations, the spatial dimensions must be divisible by
        ``2 ** 3 = 8`` for exact encoder-decoder alignment.

        Before entering the encoder, the input is padded on the bottom
        and right sides to the nearest height and width divisible by 8.

        The padding calculation is performed independently for both
        spatial dimensions:

            ``target_h = ceil(H / 8) * 8``
            ``target_w = ceil(W / 8) * 8``

        After the decoder produces the segmentation output, the padded
        regions are removed so that the final output has exactly the
        same spatial dimensions as the original input.

        This allows the model to accept arbitrary input sizes while
        preserving the original spatial dimensions in the prediction.

    Skip Connections:
        Skip connections transfer high-resolution feature maps from the
        encoder directly to the corresponding decoder stage.

        These connections help preserve spatial information that may be
        lost during max-pooling. This is especially important for
        segmentation tasks such as seismic first-break picking, where the
        precise spatial location of the target boundary is important.

    Output:
        The network returns raw, unnormalized logits with shape:

            ``(B, out_channels, H, W)``

        With the default configuration:

            ``(B, 3, H, W)``

        The model does not apply Softmax or Sigmoid to the output.
        For multi-class segmentation, the logits can be passed directly
        to ``nn.CrossEntropyLoss``.

        During inference, class labels can be obtained using
        ``torch.argmax(output, dim=1)``. Class probabilities can be
        obtained with ``torch.softmax(output, dim=1)``.

    Memory Layout:
        The final output is converted to a contiguous tensor using
        ``Tensor.contiguous()``. This provides a predictable memory layout
        and can improve compatibility with device backends and operations
        that require contiguous tensors.

    Args:
        in_channels (int, optional):
            Number of channels in the input tensor. For single-channel
            seismic data this is typically ``1``. Defaults to ``1``.

        out_channels (int, optional):
            Number of segmentation classes produced by the model.
            Defaults to ``3``.

    Attributes:
        enc1 (nn.Sequential):
            First lightweight encoder convolutional block that maps the
            input channels to 4 feature channels.

        enc2 (nn.Sequential):
            Second encoder block that maps 4 feature channels to 8.

        enc3 (nn.Sequential):
            Third encoder block that maps 8 feature channels to 16.

        pool (nn.MaxPool2d):
            Shared 2x2 max-pooling layer used for spatial downsampling
            between encoder stages.

        bottleneck (nn.Sequential):
            Deepest convolutional block, mapping 16 feature channels
            to 32.

        up3 (nn.ConvTranspose2d):
            First decoder upsampling layer, mapping 32 channels to 16.

        up2 (nn.ConvTranspose2d):
            Second decoder upsampling layer, mapping 16 channels to 8.

        up1 (nn.ConvTranspose2d):
            Final decoder upsampling layer, mapping 8 channels to 4.

        dec3 (nn.Sequential):
            First decoder convolutional block operating on the
            concatenation of the upsampled bottleneck features and
            ``enc3`` features.

        dec2 (nn.Sequential):
            Second decoder convolutional block operating on the
            concatenation of the upsampled ``dec3`` features and
            ``enc2`` features.

        dec1 (nn.Sequential):
            Final decoder convolutional block operating on the
            concatenation of the upsampled ``dec2`` features and
            ``enc1`` features.

        head (nn.Conv2d):
            1x1 convolutional output head that converts the final
            4-channel decoder representation into ``out_channels``
            segmentation logits.

    Example:
        >>> model = TinyUNet(in_channels=1, out_channels=3)
        >>> x = torch.randn(4, 1, 1578, 751)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 3, 1578, 751])

        Example of obtaining predicted segmentation classes:

        >>> predictions = torch.argmax(logits, dim=1)
        >>> predictions.shape
        torch.Size([4, 1578, 751])

        Example of using the model with CrossEntropyLoss:

        >>> criterion = nn.CrossEntropyLoss()
        >>> targets = torch.randint(0, 3, (4, 1578, 751))
        >>> loss = criterion(logits, targets)

    Notes:
        This model is intentionally designed for speed and simplicity
        rather than maximum segmentation accuracy. Its small channel
        dimensions significantly limit representational capacity compared
        with larger U-Net variants.

        Consequently, ``TinyUNet`` should primarily be considered a
        development and testing model. For final seismic first-break
        picking experiments, a larger architecture such as ``UNet`` may
        provide better feature representation and segmentation accuracy.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # Encoder (tiny channels)
        self.enc1 = self._conv_block(in_channels, 4)
        self.enc2 = self._conv_block(4, 8)
        self.enc3 = self._conv_block(8, 16)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(16, 32)

        # Decoder
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec3 = self._conv_block(32, 16)

        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec2 = self._conv_block(16, 8)

        self.up1 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.dec1 = self._conv_block(8, 4)

        self.head = nn.Conv2d(4, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # Pad to multiples of 8
        target_h = ((h + 7) // 8) * 8
        target_w = ((w + 7) // 8) * 8
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.head(d1)

        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.contiguous()
