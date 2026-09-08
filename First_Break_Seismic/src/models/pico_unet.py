"""
Pico U-Net for instant testing.
Parameters: ~2K only!
Training time: ~10 seconds per epoch
"""

import torch
import torch.nn.functional as F
from torch import nn


class PicoUNet(nn.Module):
    """
    Absolute minimal U-Net for instant testing.
    Parameters: ~2K
    Training time: ~10 seconds per epoch
    Extremely compact U-Net architecture designed for instant testing,
    debugging, prototyping, and rapid validation of segmentation pipelines.

    ``PicoUNet`` is an aggressively reduced version of the U-Net
    architecture. It preserves the fundamental encoder-decoder structure
    and skip connections while using only 1, 2, and 4 feature channels
    throughout the network. This results in an extremely small model with
    approximately 2K trainable parameters, making it suitable for tests
    where execution speed and minimal computational requirements are the
    primary objectives.

    The model is intended primarily as a development and diagnostic tool.
    It can be used to verify that the complete seismic segmentation
    pipeline works correctly before switching to larger and more
    computationally expensive architectures.

    Typical use cases include:

        - validating dataset loading and preprocessing,
        - testing tensor dimensions and data flow,
        - debugging training loops,
        - validating segmentation targets,
        - testing loss functions,
        - performing rapid overfitting tests,
        - checking optimizer and scheduler configuration,
        - validating device compatibility,
        - testing CPU/GPU/MPS execution,
        - quickly identifying errors in the training pipeline.

    Architecture:
        The network contains two encoder stages, a bottleneck, and two
        decoder stages.

        Encoder:
            - ``enc1``: ``in_channels`` -> 1 feature channel
            - ``enc2``: 1 -> 2 feature channels

        Each encoder stage consists of two 3x3 convolutional layers,
        followed by Batch Normalization and ReLU activation. A shared
        2x2 max-pooling operation reduces the spatial resolution between
        encoder stages.

        Bottleneck:
            - ``bottleneck``: 2 -> 4 feature channels

        The bottleneck represents the deepest feature representation of
        the input and operates at one-quarter of the original spatial
        resolution.

        Decoder:
            - ``up2`` / ``dec2``: 4 -> 2 feature channels
            - ``up1`` / ``dec1``: 2 -> 1 feature channel

        Each decoder stage performs learned upsampling using a transposed
        convolution. The upsampled representation is concatenated with
        the corresponding encoder feature map through a skip connection.
        The concatenated tensor is then processed by a convolutional
        block.

        Output Head:
            The final 1x1 convolution maps the single decoder feature
            channel to ``out_channels`` segmentation classes.

    Model Capacity:
        The network intentionally uses extremely small channel dimensions:

            ``1 -> 2 -> 4 -> 2 -> 1``

        This dramatically reduces the number of trainable parameters and
        computational operations compared with conventional U-Net
        architectures.

        With the default configuration, the model contains approximately
        2K trainable parameters. The exact number depends on the values of
        ``in_channels`` and ``out_channels``.

        Due to its extremely limited representational capacity, this model
        is not intended to compete with larger U-Net architectures in
        terms of final segmentation accuracy. Its primary purpose is to
        provide a fast and reliable model for development and testing.

    Input Shape:
        ``(B, C, H, W)`` where:

            - ``B`` is the batch size.
            - ``C`` is the number of input channels.
            - ``H`` is the input height.
            - ``W`` is the input width.

        For single-channel seismic data, the default input format is:

            ``(B, 1, H, W)``

        The model can accept arbitrary spatial dimensions. If the height
        or width is not divisible by 4, the input is automatically padded
        before entering the encoder.

    Padding Strategy:
        The network contains two consecutive spatial downsampling
        operations. Therefore, the input dimensions need to be divisible
        by:

            ``2 ** 2 = 4``

        To guarantee valid tensor shapes throughout the encoder and
        decoder, the ``forward`` method dynamically pads the input to the
        nearest dimensions divisible by 4.

        Padding is applied only to the bottom and right sides of the
        tensor. The original input content is therefore preserved without
        modification.

        After the decoder reconstructs the segmentation map, the padded
        regions are removed by cropping the output back to the original
        height and width.

        Consequently, the output always has the same spatial dimensions
        as the original input.

    Skip Connections:
        The model uses U-Net-style skip connections between corresponding
        encoder and decoder stages.

        The first decoder stage receives features from ``enc2`` while the
        second decoder stage receives features from ``enc1``. These
        connections preserve high-resolution spatial information that
        would otherwise be partially lost during max-pooling.

        Although the model is extremely small, skip connections retain
        the core architectural principle that makes U-Net suitable for
        spatially precise segmentation tasks.

    Output:
        The model returns raw segmentation logits with shape:

            ``(B, out_channels, H, W)``

        With the default configuration:

            ``(B, 3, H, W)``

        No Softmax or Sigmoid activation is applied to the final output.

        For multi-class segmentation, the returned logits can be passed
        directly to ``nn.CrossEntropyLoss``.

        During inference, class labels can be obtained using:

            ``torch.argmax(output, dim=1)``

        Class probabilities can be obtained using:

            ``torch.softmax(output, dim=1)``

    Seismic First-Break Picking:
        When used for seismic first-break picking, the input can represent
        a single-channel seismic section or collection of seismic traces,
        while the output represents a multi-class segmentation map.

        The extremely small architecture allows the complete first-break
        picking pipeline to be tested rapidly before using a larger model
        such as ``TinyUNet`` or ``UNet``.

        Because of its limited feature capacity, successful execution or
        even successful overfitting with ``PicoUNet`` should primarily be
        interpreted as evidence that the data pipeline, labels, loss,
        optimizer, and training procedure are functioning correctly. It
        should not necessarily be interpreted as an indication of the
        expected performance of a larger production model.

    Memory Layout:
        The final output is explicitly converted to a contiguous tensor
        using ``Tensor.contiguous()``.

        This ensures a predictable memory layout and helps maintain
        compatibility with operations and device backends that require
        contiguous tensors.

    Args:
        in_channels (int, optional):
            Number of input channels. For single-channel seismic data,
            this is typically ``1``. Defaults to ``1``.

        out_channels (int, optional):
            Number of segmentation classes produced by the network.
            Defaults to ``3``.

    Attributes:
        enc1 (nn.Sequential):
            First encoder convolutional block mapping the input to a
            single feature channel.

        enc2 (nn.Sequential):
            Second encoder convolutional block mapping 1 feature channel
            to 2 feature channels.

        pool (nn.MaxPool2d):
            Shared 2x2 max-pooling layer used for spatial downsampling.

        bottleneck (nn.Sequential):
            Deepest convolutional block mapping 2 feature channels to 4.

        up2 (nn.ConvTranspose2d):
            First decoder upsampling layer mapping 4 channels to 2.

        dec2 (nn.Sequential):
            First decoder convolutional block operating on the
            concatenation of the upsampled bottleneck representation and
            the ``enc2`` feature map.

        up1 (nn.ConvTranspose2d):
            Final decoder upsampling layer mapping 2 channels to 1.

        dec1 (nn.Sequential):
            Final decoder convolutional block operating on the
            concatenation of the upsampled representation and the
            ``enc1`` feature map.

        head (nn.Conv2d):
            1x1 convolutional output layer mapping the final single
            feature channel to ``out_channels`` segmentation classes.

    Example:
        >>> model = PicoUNet(in_channels=1, out_channels=3)
        >>> x = torch.randn(2, 1, 1578, 751)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 3, 1578, 751])

        Obtaining predicted segmentation classes:

        >>> predictions = torch.argmax(logits, dim=1)
        >>> predictions.shape
        torch.Size([2, 1578, 751])

        Using the model with CrossEntropyLoss:

        >>> criterion = nn.CrossEntropyLoss()
        >>> targets = torch.randint(0, 3, (2, 1578, 751))
        >>> loss = criterion(logits, targets)

    Notes:
        ``PicoUNet`` is intentionally optimized for execution speed and
        minimal computational cost rather than predictive performance.

        The model should primarily be used as a smoke-test or diagnostic
        architecture. If the training pipeline works correctly with
        ``PicoUNet`` but produces poor results with a larger architecture,
        the issue is more likely to be related to model capacity,
        optimization, regularization, or architecture-specific behavior
        rather than basic data-flow or tensor-shape errors.

        The approximate training time of a few seconds per epoch depends
        strongly on input size, batch size, hardware, data-loading
        overhead, and the rest of the training pipeline.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # Super tiny channels
        self.enc1 = self._conv_block(in_channels, 1)
        self.enc2 = self._conv_block(1, 2)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(2, 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(4, 2, 2, stride=2)
        self.dec2 = self._conv_block(4, 2)

        self.up1 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        self.dec1 = self._conv_block(2, 1)

        self.head = nn.Conv2d(1, out_channels, kernel_size=1)

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

        # Pad to multiples of 4
        target_h = ((h + 3) // 4) * 4
        target_w = ((w + 3) // 4) * 4
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)  # 1 channel
        p1 = self.pool(e1)  # half size
        e2 = self.enc2(p1)  # 2 channels
        p2 = self.pool(e2)  # quarter size
        b = self.bottleneck(p2)  # 4 channels

        # Decoder with skip connections
        d2 = self.up2(b)  # 2 channels
        d2 = torch.cat([d2, e2], dim=1)  # 4 channels
        d2 = self.dec2(d2)  # 2 channels

        d1 = self.up1(d2)  # 1 channel
        d1 = torch.cat([d1, e1], dim=1)  # 2 channels
        d1 = self.dec1(d1)  # 1 channel

        out = self.head(d1)  # 3 channels

        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.contiguous()
