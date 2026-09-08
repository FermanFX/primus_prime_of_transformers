"""
MPS-optimized lightweight U-Net for seismic segmentation.
Specifically designed for Apple Silicon MPS memory constraints.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MPSLightUNet(nn.Module):
    """
    Lightweight U-Net optimized for MPS memory.

    Parameters: ~1.7M
    Memory: ~100 MB
    Speed: ~2× faster than original UNet

    Lightweight U-Net architecture optimized for memory-constrained
    execution on Apple's Metal Performance Shaders (MPS) backend.

    ``MPSLightUNet`` is a reduced-capacity variant of the conventional
    U-Net architecture designed specifically for efficient training and
    inference on Apple Silicon devices. It preserves the main architectural
    characteristics of U-Net, including hierarchical feature extraction,
    encoder-decoder processing, transposed-convolution upsampling, and
    skip connections, while substantially reducing the number of feature
    channels used at each stage.

    The reduced channel configuration lowers the model's parameter count,
    intermediate activation sizes, and overall memory requirements. This
    makes the architecture particularly useful when training seismic
    segmentation models on Apple Silicon systems where MPS memory usage
    can become a practical constraint.

    The model is intended for multi-class semantic segmentation of
    single-channel seismic data, including seismic first-break picking.
    Each spatial location of the input is mapped to one of
    ``out_channels`` segmentation classes.

    Architecture:
        The network consists of four encoder stages, a bottleneck, and four
        decoder stages.

        Encoder:
            - ``enc1``: ``in_channels`` -> 16 feature channels
            - ``enc2``: 16 -> 32 feature channels
            - ``enc3``: 32 -> 64 feature channels
            - ``enc4``: 64 -> 128 feature channels

        Each encoder stage contains two 3x3 convolutional layers. Every
        convolution is followed by Batch Normalization and ReLU activation.

        A shared 2x2 max-pooling layer performs spatial downsampling between
        encoder stages. Four pooling operations reduce the spatial
        resolution by a factor of 16 at the bottleneck.

        Bottleneck:
            - ``bottleneck``: 128 -> 256 feature channels

        The bottleneck provides the deepest feature representation while
        operating at the lowest spatial resolution in the network.

        Decoder:
            - ``up4`` / ``dec4``: 256 -> 128 feature channels
            - ``up3`` / ``dec3``: 128 -> 64 feature channels
            - ``up2`` / ``dec2``: 64 -> 32 feature channels
            - ``up1`` / ``dec1``: 32 -> 16 feature channels

        Each decoder stage begins with a 2x2 transposed convolution with
        stride 2, doubling the spatial resolution.

        The upsampled feature representation is concatenated with the
        corresponding encoder feature map through a skip connection. The
        concatenated tensor is subsequently processed by a convolutional
        block.

        Output Head:
            A final 1x1 convolution maps the 16-channel decoder output to
            ``out_channels`` segmentation classes.

    MPS Optimization:
        The architecture is specifically designed to reduce memory
        consumption on Apple's MPS backend.

        Compared with a conventional U-Net using substantially larger
        channel dimensions, this model uses the following progression:

            ``16 -> 32 -> 64 -> 128 -> 256``

        and then symmetrically reduces the number of channels through the
        decoder.

        Reducing the number of channels decreases both trainable parameter
        count and the size of intermediate feature maps. This is especially
        useful for MPS workloads because activation tensors generated
        during forward and backward passes can contribute significantly
        to peak memory consumption.

        The final output is also explicitly converted to a contiguous
        tensor using ``Tensor.contiguous()`` to provide a predictable
        memory layout for subsequent operations and improve compatibility
        with device-specific tensor requirements.

    Model Capacity:
        The model contains approximately 1.7 million trainable parameters
        with the default input and output channel configuration.

        This represents a substantial reduction in model capacity compared
        with a larger conventional U-Net while retaining significantly
        more representational power than extremely lightweight variants
        such as ``PicoUNet`` or ``NanoUNet``.

        The architecture therefore provides a practical compromise between
        segmentation capability and memory consumption.

    Input Shape:
        ``(B, C, H, W)`` where:

            - ``B`` is the batch size.
            - ``C`` is the number of input channels.
            - ``H`` is the spatial height.
            - ``W`` is the spatial width.

        For the default single-channel seismic configuration:

            ``(B, 1, H, W)``

        The model accepts arbitrary spatial dimensions. Dimensions that
        are not divisible by 16 are automatically padded before entering
        the encoder.

    Padding Strategy:
        Because the architecture performs four consecutive 2x2 pooling
        operations, the input spatial dimensions must be divisible by:

            ``2 ** 4 = 16``

        for exact encoder-decoder spatial alignment.

        The ``forward`` method therefore computes the smallest height and
        width greater than or equal to the original dimensions that are
        divisible by 16.

        Padding is applied only to the bottom and right sides:

            ``F.pad(x, (0, pad_w, 0, pad_h))``

        This preserves the original seismic data and avoids changing the
        existing spatial coordinates.

        After decoding, the additional padded region is removed by
        cropping the output back to the original height and width.

        Therefore, regardless of whether padding was required, the final
        prediction has the same spatial dimensions as the original input.

    Skip Connections:
        The model uses four encoder-decoder skip connections:

            - ``e4`` -> ``dec4``
            - ``e3`` -> ``dec3``
            - ``e2`` -> ``dec2``
            - ``e1`` -> ``dec1``

        These connections preserve high-resolution spatial information
        extracted before each downsampling operation.

        Skip connections are particularly important for seismic
        segmentation because first-break events can contain fine-scale
        spatial structures that may be difficult to reconstruct from the
        low-resolution bottleneck representation alone.

    Output:
        The model produces raw segmentation logits with shape:

            ``(B, out_channels, H, W)``

        With the default configuration:

            ``(B, 3, H, W)``

        The output does not contain a Softmax or Sigmoid activation.

        For multi-class segmentation, the raw logits can be passed
        directly to ``nn.CrossEntropyLoss``.

        During inference, class predictions can be obtained with:

            ``torch.argmax(output, dim=1)``

        Class probabilities can be obtained using:

            ``torch.softmax(output, dim=1)``

    Seismic First-Break Picking:
        ``MPSLightUNet`` is suitable for seismic first-break picking when
        the task is formulated as multi-class semantic segmentation.

        A seismic section can be represented as a single-channel 2D input,
        while the model produces a class prediction for every spatial
        location.

        The architecture is particularly useful for experiments performed
        on Apple Silicon hardware because it provides substantially more
        feature capacity than ultra-small testing models while keeping
        memory requirements considerably lower than a full-sized U-Net.

        This makes it suitable for the following workflow:

            ``PicoUNet``
                ->
            ``NanoUNet``
                ->
            ``TinyUNet``
                ->
            ``MPSLightUNet``
                ->
            ``UNet``

        Smaller models can first be used to validate the training pipeline,
        after which ``MPSLightUNet`` can provide a more capable model while
        remaining relatively lightweight for MPS execution.

    Memory Considerations:
        The model is designed around an approximate memory footprint of
        100 MB under the intended workload. Actual memory usage can vary
        significantly depending on input dimensions, batch size, tensor
        precision, optimizer state, gradients, PyTorch version, MPS
        behavior, and other components of the training pipeline.

        The stated memory and speed characteristics should therefore be
        treated as approximate benchmarks rather than guaranteed values.

        Peak training memory is generally higher than inference memory
        because intermediate activations and gradients must be retained
        during backpropagation.

    Performance:
        The architecture is expected to provide substantially lower
        computational and memory cost than the original full-sized U-Net.

        A nominal speed improvement of approximately 2x may be observed
        under comparable experimental conditions, but actual performance
        depends on hardware, batch size, input resolution, data-loading
        overhead, precision, and the PyTorch/MPS software stack.

    Args:
        in_channels (int, optional):
            Number of channels in the input tensor. For single-channel
            seismic data this is typically ``1``. Defaults to ``1``.

        out_channels (int, optional):
            Number of segmentation classes produced by the model.
            Defaults to ``3``.

    Attributes:
        enc1 (nn.Sequential):
            First encoder convolutional block mapping the input to
            16 feature channels.

        enc2 (nn.Sequential):
            Second encoder convolutional block mapping 16 channels to 32.

        enc3 (nn.Sequential):
            Third encoder convolutional block mapping 32 channels to 64.

        enc4 (nn.Sequential):
            Fourth encoder convolutional block mapping 64 channels to 128.

        pool (nn.MaxPool2d):
            Shared 2x2 max-pooling layer used for spatial downsampling.

        bottleneck (nn.Sequential):
            Deepest convolutional block mapping 128 feature channels
            to 256.

        up4 (nn.ConvTranspose2d):
            First decoder upsampling layer mapping 256 channels to 128.

        dec4 (nn.Sequential):
            First decoder convolutional block processing the concatenation
            of the upsampled bottleneck representation and ``enc4``.

        up3 (nn.ConvTranspose2d):
            Second decoder upsampling layer mapping 128 channels to 64.

        dec3 (nn.Sequential):
            Second decoder convolutional block processing the concatenation
            of the upsampled ``dec4`` representation and ``enc3``.

        up2 (nn.ConvTranspose2d):
            Third decoder upsampling layer mapping 64 channels to 32.

        dec2 (nn.Sequential):
            Third decoder convolutional block processing the concatenation
            of the upsampled ``dec3`` representation and ``enc2``.

        up1 (nn.ConvTranspose2d):
            Final decoder upsampling layer mapping 32 channels to 16.

        dec1 (nn.Sequential):
            Final decoder convolutional block processing the concatenation
            of the upsampled ``dec2`` representation and ``enc1``.

        head (nn.Conv2d):
            1x1 convolutional output head mapping the final 16-channel
            representation to ``out_channels`` segmentation logits.

    Example:
        >>> model = MPSLightUNet(in_channels=1, out_channels=3)
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

        Moving the model and input to Apple Silicon MPS:

        >>> device = torch.device("mps")
        >>> model = MPSLightUNet().to(device)
        >>> x = torch.randn(2, 1, 1578, 751, device=device)
        >>> logits = model(x)

    Notes:
        ``MPSLightUNet`` is designed as a practical memory-efficient
        alternative to a full-sized U-Net rather than as a strict
        performance replacement.

        The approximate parameter count, memory usage, and speed
        characteristics depend on the exact input configuration and
        execution environment.

        For very fast pipeline validation, ``PicoUNet`` or ``NanoUNet``
        may be preferable. For higher representational capacity,
        ``MPSLightUNet`` provides a stronger intermediate architecture
        before moving to a full-sized ``UNet``.
    
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()

        # --- Encoder (reduced channels) ---
        self.enc1 = self._conv_block(in_channels, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = self._conv_block(128, 256)

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = self._conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._conv_block(32, 16)

        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

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

        # Pad to multiples of 16 for clean pooling/unpooling
        target_h = ((h + 15) // 16) * 16
        target_w = ((w + 15) // 16) * 16
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
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
