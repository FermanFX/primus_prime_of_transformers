"""
Nano U-Net for ultra-fast testing.
Parameters: ~10K only!
Training time: ~30 seconds per epoch
"""

import torch
import torch.nn.functional as F
from torch import nn


class NanoUNet(nn.Module):
    """
    Ultra-lightweight U-Net for quick testing.
    Parameters: ~10K
    Training time: ~2 min per epoch
        Ultra-lightweight U-Net architecture designed for fast testing,
    debugging, prototyping, and rapid validation of semantic segmentation
    pipelines.

    ``NanoUNet`` provides a lightweight compromise between the extremely
    small ``PicoUNet`` and the larger ``TinyUNet`` architecture. It
    preserves the fundamental U-Net encoder-decoder structure, including
    convolutional feature extraction, spatial downsampling, learned
    upsampling, and encoder-decoder skip connections, while maintaining a
    very small number of feature channels.

    The architecture uses 2, 4, 8, and 16 feature channels at its
    successive encoder and bottleneck stages. This results in a model with
    approximately 10K trainable parameters, making it suitable for
    experiments that require significantly faster execution than a
    conventional U-Net.

    The primary purpose of this model is not maximum predictive accuracy,
    but rapid verification of machine-learning pipelines. It can be used
    to determine whether data loading, preprocessing, tensor dimensions,
    segmentation labels, loss functions, optimizers, and training loops
    are functioning correctly before moving to a larger architecture.

    Typical use cases include:

        - validating seismic data preprocessing,
        - testing dataset and dataloader implementations,
        - checking input and target tensor dimensions,
        - debugging segmentation losses,
        - validating training and validation loops,
        - performing quick overfitting experiments,
        - testing optimizer configurations,
        - testing learning-rate schedules,
        - validating CPU/GPU/MPS execution,
        - checking model serialization and loading,
        - rapidly comparing different training configurations,
        - smoke-testing a complete seismic first-break picking pipeline.

    Architecture:
        The model contains three encoder stages, one bottleneck, and three
        decoder stages.

        Encoder:
            - ``enc1``: ``in_channels`` -> 2 feature channels
            - ``enc2``: 2 -> 4 feature channels
            - ``enc3``: 4 -> 8 feature channels

        Each encoder stage contains two 3x3 convolutional layers. Each
        convolution is followed by Batch Normalization and ReLU
        activation.

        A shared 2x2 max-pooling layer is applied between encoder stages,
        reducing the spatial resolution by a factor of two at each level.

        Bottleneck:
            - ``bottleneck``: 8 -> 16 feature channels

        The bottleneck represents the deepest feature representation of
        the input and operates at one-eighth of the original spatial
        resolution.

        Decoder:
            - ``up3`` / ``dec3``: 16 -> 8 feature channels
            - ``up2`` / ``dec2``: 8 -> 4 feature channels
            - ``up1`` / ``dec1``: 4 -> 2 feature channels

        Each decoder stage begins with a transposed convolution that
        doubles the spatial resolution.

        The upsampled feature map is concatenated with the corresponding
        encoder feature map through a skip connection. The concatenated
        representation is then processed by a convolutional block.

        Output Head:
            A final 1x1 convolution maps the final 2-channel decoder
            representation to ``out_channels`` segmentation classes.

    Model Capacity:
        The model intentionally uses a small channel progression:

            ``2 -> 4 -> 8 -> 16 -> 8 -> 4 -> 2``

        This provides more representational capacity than ``PicoUNet``
        while remaining substantially smaller than standard U-Net
        architectures.

        With the default configuration, the model contains approximately
        10K trainable parameters. The exact number depends on the values of
        ``in_channels`` and ``out_channels``.

        The reduced model capacity provides a useful balance between
        computational efficiency and the ability to learn meaningful
        spatial features.

    Input Shape:
        ``(B, C, H, W)`` where:

            - ``B`` is the batch size.
            - ``C`` is the number of input channels.
            - ``H`` is the spatial height.
            - ``W`` is the spatial width.

        For single-channel seismic data, the default input format is:

            ``(B, 1, H, W)``

        The model accepts arbitrary spatial dimensions. Input dimensions
        that are not divisible by 8 are automatically padded before the
        encoder.

    Padding Strategy:
        The network performs three consecutive spatial downsampling
        operations. Therefore, the spatial dimensions need to be divisible
        by:

            ``2 ** 3 = 8``

        To ensure that the encoder and decoder feature maps can be aligned
        correctly, the ``forward`` method pads the input to the nearest
        height and width divisible by 8.

        Padding is applied only to the bottom and right sides of the input.
        This preserves the original seismic data without changing or
        shifting the existing spatial information.

        After the decoder reconstructs the segmentation map, the padded
        regions are removed by cropping the output to the original input
        dimensions.

        As a result, the final output always has the same spatial height
        and width as the original input.

    Skip Connections:
        The architecture uses three U-Net-style skip connections between
        corresponding encoder and decoder stages.

        Specifically:

            - ``e3`` is connected to the first decoder stage,
            - ``e2`` is connected to the second decoder stage,
            - ``e1`` is connected to the final decoder stage.

        These connections provide the decoder with high-resolution spatial
        information that may have been lost during the pooling operations.

        This is particularly useful for seismic segmentation and
        first-break picking, where accurate localization of boundaries and
        events is often more important than coarse semantic classification.

    Output:
        The model returns raw, unnormalized segmentation logits with shape:

            ``(B, out_channels, H, W)``

        With the default configuration:

            ``(B, 3, H, W)``

        The output does not contain a Softmax or Sigmoid activation.

        For multi-class semantic segmentation, the logits can therefore
        be passed directly to ``nn.CrossEntropyLoss``.

        During inference, the predicted class at each spatial location
        can be obtained with:

            ``torch.argmax(output, dim=1)``

        Class probabilities can be obtained using:

            ``torch.softmax(output, dim=1)``

    Seismic First-Break Picking:
        ``NanoUNet`` can be used as a lightweight baseline for seismic
        first-break picking formulated as a multi-class segmentation
        problem.

        The model can process a single-channel seismic section and produce
        a class prediction for each spatial location. Its low computational
        cost makes it useful for rapidly testing whether first-break
        labels, preprocessing, and the training pipeline are internally
        consistent.

        However, due to its limited number of feature channels, the model
        has substantially less representational capacity than a full-sized
        U-Net. Performance obtained with this model should therefore be
        interpreted primarily as a pipeline validation result rather than
        an estimate of the final achievable segmentation performance.

    Comparison with Smaller and Larger Variants:
        ``NanoUNet`` occupies an intermediate position between extremely
        minimal and full-sized U-Net models.

        Conceptually, the progression is:

            ``PicoUNet`` -> ``NanoUNet`` -> ``TinyUNet`` -> ``UNet``

        ``PicoUNet`` prioritizes the lowest possible computational cost,
        while ``UNet`` provides substantially greater feature capacity.
        ``NanoUNet`` offers a middle ground that can be useful when
        ``PicoUNet`` is too restrictive but a larger ``TinyUNet`` or
        ``UNet`` is unnecessarily expensive for an initial experiment.

    Memory Layout:
        The final output is explicitly converted to a contiguous tensor
        using ``Tensor.contiguous()``.

        This ensures a predictable memory layout and improves compatibility
        with operations or device backends that require contiguous tensor
        storage.

    Args:
        in_channels (int, optional):
            Number of channels in the input tensor. For single-channel
            seismic data, this is typically ``1``. Defaults to ``1``.

        out_channels (int, optional):
            Number of segmentation classes produced by the network.
            Defaults to ``3``.

    Attributes:
        enc1 (nn.Sequential):
            First encoder convolutional block mapping the input channels
            to 2 feature channels.

        enc2 (nn.Sequential):
            Second encoder convolutional block mapping 2 feature channels
            to 4.

        enc3 (nn.Sequential):
            Third encoder convolutional block mapping 4 feature channels
            to 8.

        pool (nn.MaxPool2d):
            Shared 2x2 max-pooling layer used for spatial downsampling.

        bottleneck (nn.Sequential):
            Deepest convolutional block mapping 8 feature channels to 16.

        up3 (nn.ConvTranspose2d):
            First decoder upsampling layer mapping 16 channels to 8.

        dec3 (nn.Sequential):
            First decoder convolutional block operating on the concatenated
            output of ``up3`` and the ``enc3`` feature map.

        up2 (nn.ConvTranspose2d):
            Second decoder upsampling layer mapping 8 channels to 4.

        dec2 (nn.Sequential):
            Second decoder convolutional block operating on the
            concatenated output of ``up2`` and the ``enc2`` feature map.

        up1 (nn.ConvTranspose2d):
            Final decoder upsampling layer mapping 4 channels to 2.

        dec1 (nn.Sequential):
            Final decoder convolutional block operating on the
            concatenated output of ``up1`` and the ``enc1`` feature map.

        head (nn.Conv2d):
            1x1 convolutional output head mapping the final 2 decoder
            feature channels to ``out_channels`` segmentation logits.

    Example:
        >>> model = NanoUNet(in_channels=1, out_channels=3)
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
        ``NanoUNet`` is intended primarily for rapid experimentation and
        pipeline validation. Its approximately 10K parameters allow it to
        train considerably faster than larger U-Net variants.

        Actual training time depends on input dimensions, batch size,
        hardware, data-loading overhead, precision, and the surrounding
        training pipeline. The stated training time of approximately
        30 seconds per epoch should therefore be treated as an
        environment-dependent estimate rather than a guaranteed runtime.

        If ``NanoUNet`` successfully trains and overfits a small subset of
        the data, this can provide useful evidence that the basic data,
        labels, loss function, and training loop are working correctly.
        Further improvements in segmentation quality may then require
        increasing model capacity or changing the architecture.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # Encoder (tiny channels)
        self.enc1 = self._conv_block(in_channels, 2)
        self.enc2 = self._conv_block(2, 4)
        self.enc3 = self._conv_block(4, 8)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(8, 16)

        # Decoder
        self.up3 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec3 = self._conv_block(16, 8)

        self.up2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.dec2 = self._conv_block(8, 4)

        self.up1 = nn.ConvTranspose2d(4, 2, 2, stride=2)
        self.dec1 = self._conv_block(4, 2)

        self.head = nn.Conv2d(2, out_channels, kernel_size=1)

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
