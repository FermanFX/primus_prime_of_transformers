"""
Configuration management for the Seismic FBP pipeline.

This module defines the :class:`SeismicConfig` dataclass, which stores,
validates, and serializes all configuration parameters required by the
seismic first-break picking (FBP) pipeline.

The configuration includes dataset processing, training, loss functions,
learning-rate scheduling, regularization, early stopping, logging, and
debugging options.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SeismicConfig:
    """Central configuration for the seismic FBP training pipeline.

    This class provides a single, validated configuration object for
    dataset preprocessing, model training, loss calculation, learning-rate
    scheduling, checkpoint management, logging, and debugging.

    Configuration values are validated automatically during initialization.
    Required output directories are also created automatically.

    Attributes:
        dataset_name: Name of the seismic dataset.
        hdf5_path: Path to the source HDF5 dataset.
        chunk_dir: Directory containing preprocessed data chunks.
        preprocess: Whether preprocessing should be performed.
        force_reprocess: Whether existing preprocessed chunks should be
            regenerated.

        target_traces: Number of seismic traces used by the pipeline.
        n_samples: Number of samples per seismic trace.
        strip_width: Width of the seismic strip used for processing.
            Must be a positive even number.
        chunk_size: Number of samples/traces processed in a chunk.
        random_seed: Random seed used for reproducibility.
        train_split: Fraction of data allocated to training.
        val_split: Fraction of data allocated to validation.
        test_split: Fraction of data allocated to testing.

        batch_size: Number of samples processed in each training batch.
        learning_rate: Initial optimizer learning rate.
        n_epochs: Maximum number of training epochs.
        device: Computation device. Supported values are ``"cpu"``,
            ``"cuda"``, and ``"mps"``.
        num_workers: Number of worker processes used by data loaders.
        multi_gpu: Whether multi-GPU training is enabled.
        gpu_ids: Optional list of GPU device IDs.

        class_weights: Per-class weights used by the classification loss.

        model_registry_dir: Directory used for model registry metadata.
        checkpoint_dir: Directory used to store training checkpoints.
        checkpoint_every: Frequency, in epochs, for saving checkpoints.

        cache_size: Maximum number of data chunks kept in memory.

        lr_scheduler: Learning-rate scheduler type. Supported values are
            ``"step"``, ``"plateau"``, and ``"cosine"``.
        lr_patience: Number of epochs without improvement before reducing
            the learning rate when using the plateau scheduler.
        lr_factor: Multiplicative factor used when reducing the learning
            rate.
        lr_step_size: Number of epochs between learning-rate reductions
            for the step scheduler.
        lr_gamma: Multiplicative factor used by the step scheduler.
        lr_T_max: Maximum number of iterations/epochs for the cosine
            scheduler.

        loss_function: Name of the loss function used during training.
        dice_weight: Weight assigned to the Dice component of the loss.
        focal_gamma: Focusing parameter used by focal loss.

        gradient_clip_value: Maximum gradient norm used for gradient
            clipping. ``None`` disables gradient clipping.

        early_stopping_patience: Number of epochs without sufficient
            improvement before stopping training. ``None`` disables early
            stopping.
        early_stopping_min_delta: Minimum change considered an improvement
            for early stopping.

        tensorboard_log_dir: Directory for TensorBoard logs.
        mlflow_experiment_name: Name of the MLflow experiment.
        log_dir: Directory for application log files.
        log_level: Logging verbosity level.
        log_memory: Whether memory usage should be logged.
        log_predictions_every: Frequency, in epochs, for logging model
            predictions.
        log_metrics_every: Frequency, in epochs, for logging metrics.
        log_gradients: Whether model gradients should be logged.

        verbose_training: Whether detailed training information is printed.
        log_batch_every: Frequency, in batches, for batch-level logging.
            ``None`` disables batch-level logging.

    Raises:
        ValueError: If any configuration parameter is invalid.
    """

    # === Dataset ===
    dataset_name: str = "Halfmile"
    hdf5_path: str = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
    chunk_dir: str = "data/chunks"
    preprocess: bool = False
    force_reprocess: bool = False

    # === Data ===
    target_traces: int = 1578
    n_samples: int = 751
    strip_width: int = 8
    chunk_size: int = 69
    random_seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # === Training ===
    batch_size: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 30
    device: str = "mps"
    num_workers: int = 0
    multi_gpu: bool = False
    gpu_ids: list | None = None

    # === Loss ===
    class_weights: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.6])

    # === Model Registry ===
    model_registry_dir: str = "models/registry"
    checkpoint_dir: str = "models/registry"
    checkpoint_every: int = 5

    # === Cache ===
    cache_size: int = 3  # Number of chunks to keep in memory

    # === Scheduler ===
    lr_scheduler: str = "plateau"
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    lr_T_max: int = 30

    loss_function: str = "cross_entropy"
    dice_weight: float = 0.5
    focal_gamma: float = 2.0

    # === Regularization ===
    gradient_clip_value: float | None = 1.0

    # === Early Stopping ===
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 1e-4

    # === Logging ===
    tensorboard_log_dir: str = "runs"
    mlflow_experiment_name: str = "seismic-fbp"
    log_dir: str = "logs"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_memory: bool = False
    log_predictions_every: int = 5
    log_metrics_every: int = 1
    log_gradients: bool = False

    # === Debugging ===
    verbose_training: bool = False
    log_batch_every: int | None = None  # None = disabled

    def __post_init__(self):
        """Validate configuration and create required directories.

        This method is called automatically by ``dataclass`` after object
        initialization. It validates dataset dimensions, train/validation/
        test split ratios, training parameters, loss weights, logging level,
        scheduler type, and computation device.

        Raises:
            ValueError: If one or more configuration values are invalid.
        """
        # Data validation
        if self.target_traces <= 0:
            raise ValueError(
                f"target_traces must be positive, got {self.target_traces}"
            )
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.strip_width <= 0 or self.strip_width % 2 != 0:
            raise ValueError(
                f"strip_width must be positive and even, got {self.strip_width}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")

        # Split validation
        if not abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-9:
            raise ValueError(
                f"train+val+test split must equal 1.0, "
                f"got {self.train_split + self.val_split + self.test_split}"
            )

        # Training validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {self.num_workers}"
            )

        # Loss validation
        if len(self.class_weights) != 3:
            raise ValueError(
                f"class_weights must have exactly 3 values, got {len(self.class_weights)}"
            )
        if any(w < 0 for w in self.class_weights):
            raise ValueError(
                f"class_weights must be non-negative, got {self.class_weights}"
            )

        # Log level validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got {self.log_level}"
            )

        # Scheduler validation
        valid_schedulers = ["step", "plateau", "cosine"]
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, got {self.lr_scheduler}"
            )

        # Device validation
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got {self.device}"
            )

        # Create directories
        Path(self.model_registry_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def get_config_hash(self) -> str:
        """Generate a short deterministic hash of key configuration values.

        The hash is useful for identifying experiments that use different
        model-training or dataset parameters.

        Returns:
            An 8-character hexadecimal MD5 hash.
        """
        config_dict = {
            "dataset_name": self.dataset_name,
            "target_traces": self.target_traces,
            "n_samples": self.n_samples,
            "strip_width": self.strip_width,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler,
            "class_weights": self.class_weights,
        }
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary.

        Private attributes, if any, are excluded from the resulting
        dictionary.

        Returns:
            Dictionary containing all public configuration values.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_yaml(self) -> str:
        """Serialize the configuration to a YAML-formatted string.

        Returns:
            YAML representation of the current configuration.
        """
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        """Return a human-readable representation of the configuration.

        Returns:
            A string containing the class name and all public configuration
            values.
        """
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"SeismicConfig({', '.join(items)})"
