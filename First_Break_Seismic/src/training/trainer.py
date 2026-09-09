"""
Training loop with MLflow, TensorBoard, and Loguru integration.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import SeismicConfig
from src.training.metrics import (
    SegmentationMetrics,
    compute_gradient_norm,
    compute_weight_norm,
)
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLflowManager


logger = get_logger()


class SeismicTrainer:
    """
    Integrated trainer with MLflow, TensorBoard, and Loguru.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SeismicConfig,
        model_name: str = "unet",
    ):
        self.config = config
        self.model_name = model_name
        self.device = self._setup_device(config.device)

        # Setup model
        self.model: nn.Module

        if config.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model, device_ids=config.gpu_ids)
            logger.info(
                f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs"
            )
        else:
            self.model = model
            logger.info(f"Single device: {self.device}")

        self.model = self.model.to(self.device)
        self.dataloaders = dataloaders
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = self._create_scheduler()

        # Model Registry Directory
        self.registry_dir = Path(config.model_registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb_dir = (
            Path(config.tensorboard_log_dir)
            / config.dataset_name
            / model_name
        )
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        # MLflow
        self.mlflow_manager = MLflowManager(
            experiment_name=config.mlflow_experiment_name,
            enable_system_metrics=True,
            enable_autolog=True,
        )

        self.registered_models: dict[str, Any] = {}

        logger.info(f"Model registry: {self.registry_dir}")
        logger.info(f"TensorBoard: {self.tb_dir}")

    def _setup_device(self, requested_device: str) -> torch.device:
        """Setup device with fallback."""
        if requested_device == "mps":
            if torch.backends.mps.is_available():
                try:
                    test = torch.ones(1, device="mps")
                    del test

                    logger.info("MPS device initialized successfully")
                    return torch.device("mps")

                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"MPS initialization failed: {e}, "
                        "falling back to CPU"
                    )
                    return torch.device("cpu")

            logger.warning("MPS not available, falling back to CPU")
            return torch.device("cpu")

        if requested_device == "cuda":
            if torch.cuda.is_available():
                logger.info(
                    "CUDA device initialized: "
                    f"{torch.cuda.get_device_name(0)}"
                )
                return torch.device("cuda")

            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

        return torch.device("cpu")

    def _create_scheduler(
        self,
    ) -> (
        torch.optim.lr_scheduler.StepLR
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | torch.optim.lr_scheduler.CosineAnnealingLR
        | None
    ):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )

        if self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.lr_patience,
                factor=self.config.lr_factor,
            )

        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_T_max,
            )

        return None

    def _criterion_returns_components(self) -> bool:
        """
        Safely determine whether the criterion returns loss components.

        Keeping this check in a dedicated method prevents mypy from
        confusing the dynamically accessed attribute with nn.Module
        attributes/types.
        """
        return cast(
            bool,
            getattr(self.criterion, "return_components", False),
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint with full state restoration."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        target_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )

        try:
            target_model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            logger.info("Model state loaded successfully")

        except RuntimeError as e:
            logger.warning(
                f"Model state mismatch: {e}, "
                "loading with strict=False"
            )
            target_model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=False,
            )

        try:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            logger.info("Optimizer state loaded successfully")

        except ValueError as e:
            logger.warning(
                f"Optimizer state mismatch: {e}, "
                "using fresh optimizer state"
            )

        if (
            self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
        ):
            try:
                self.scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )
                logger.info("Scheduler state loaded successfully")

            except ValueError as e:
                logger.warning(
                    f"Scheduler state mismatch: {e}, "
                    "using fresh scheduler state"
                )

        logger.info(
            "Resumed from epoch "
            f"{checkpoint['epoch']} with "
            f"val_loss={checkpoint.get('val_loss', 'N/A')}"
        )

        return int(checkpoint["epoch"])

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage without sudo."""
        memory: dict[str, float] = {}

        if self.device.type == "mps":
            memory["allocated"] = (
                torch.mps.current_allocated_memory() / 1e9
            )
            memory["max_allocated"] = (
                torch.mps.driver_allocated_memory() / 1e9
            )

        elif self.device.type == "cuda":
            memory["allocated"] = (
                torch.cuda.memory_allocated() / 1e9
            )
            memory["reserved"] = (
                torch.cuda.memory_reserved() / 1e9
            )
            memory["max_allocated"] = (
                torch.cuda.max_memory_allocated() / 1e9
            )

        return memory

    def train_epoch(
        self,
        verbose: bool = False,
    ) -> tuple[float, dict]:
        """
        Run one training epoch.

        Returns:
            avg_loss: Average training loss.
            metrics: Dictionary of training metrics.
        """
        self.model.train()

        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3)

        if self.device.type == "mps":
            torch.mps.empty_cache()

        loss_components: dict[str, Any] = {
            "total": 0.0,
            "ce": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "ce_focal_combined": 0.0,
            "per_class": {},
        }

        num_batches = 0

        train_loader = self.dataloaders["train"]

        pbar = tqdm(
            train_loader,
            desc="Training",
        )

        criterion_has_components = (
            self._criterion_returns_components()
        )

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(
                self.device,
                non_blocking=True,
            )
            y = y.to(
                self.device,
                non_blocking=True,
            )

            x = x.contiguous()
            y = y.contiguous()

            self.optimizer.zero_grad()

            outputs: torch.Tensor = self.model(x)

            loss: torch.Tensor

            if criterion_has_components:
                loss, components_raw = self.criterion(
                    outputs,
                    y,
                )

                components = cast(
                    dict[str, Any],
                    components_raw,
                )

                loss_components["total"] += float(
                    components["total"]
                )
                loss_components["ce"] += float(
                    components["ce"]
                )
                loss_components["focal"] += float(
                    components["focal"]
                )
                loss_components["dice"] += float(
                    components["dice"]
                )
                loss_components["ce_focal_combined"] += float(
                    components["ce_focal_combined"]
                )

                per_class = cast(
                    dict[str, float],
                    components["per_class"],
                )

                for class_name, class_loss in per_class.items():
                    if (
                        class_name
                        not in loss_components["per_class"]
                    ):
                        loss_components["per_class"][
                            class_name
                        ] = 0.0

                    loss_components["per_class"][
                        class_name
                    ] += float(class_loss)

            else:
                loss = self.criterion(outputs, y)

            loss.backward()

            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            seg_metrics.update(
                preds,
                y,
            )

            if verbose and batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}/"
                    f"{len(train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                }
            )

        if num_batches == 0:
            raise RuntimeError(
                "Training dataloader is empty."
            )

        avg_loss = total_loss / num_batches
        metrics = seg_metrics.compute()

        if criterion_has_components:
            for key in (
                "total",
                "ce",
                "focal",
                "dice",
                "ce_focal_combined",
            ):
                metrics[f"loss_{key}"] = (
                    loss_components[key] / num_batches
                )

            per_class_total = cast(
                dict[str, float],
                loss_components["per_class"],
            )

            for class_name, class_loss in (
                per_class_total.items()
            ):
                metrics[f"loss_{class_name}"] = (
                    class_loss / num_batches
                )

        return avg_loss, metrics

    @torch.no_grad()
    def validate(
        self,
        verbose: bool = False,
    ) -> tuple[float, dict]:
        """
        Run validation.

        Returns:
            avg_loss: Average validation loss.
            metrics: Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3)

        if self.device.type == "mps":
            torch.mps.empty_cache()

        loss_components: dict[str, Any] = {
            "total": 0.0,
            "ce": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "ce_focal_combined": 0.0,
            "per_class": {},
        }

        num_batches = 0

        val_loader = self.dataloaders["val"]

        pbar = tqdm(
            val_loader,
            desc="Validation",
        )

        criterion_has_components = (
            self._criterion_returns_components()
        )

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(
                self.device,
                non_blocking=True,
            )
            y = y.to(
                self.device,
                non_blocking=True,
            )

            x = x.contiguous()
            y = y.contiguous()

            outputs: torch.Tensor = self.model(x)

            loss: torch.Tensor

            if criterion_has_components:
                loss, components_raw = self.criterion(
                    outputs,
                    y,
                )

                components = cast(
                    dict[str, Any],
                    components_raw,
                )

                loss_components["total"] += float(
                    components["total"]
                )
                loss_components["ce"] += float(
                    components["ce"]
                )
                loss_components["focal"] += float(
                    components["focal"]
                )
                loss_components["dice"] += float(
                    components["dice"]
                )
                loss_components["ce_focal_combined"] += float(
                    components["ce_focal_combined"]
                )

                per_class = cast(
                    dict[str, float],
                    components["per_class"],
                )

                for class_name, class_loss in per_class.items():
                    if (
                        class_name
                        not in loss_components["per_class"]
                    ):
                        loss_components["per_class"][
                            class_name
                        ] = 0.0

                    loss_components["per_class"][
                        class_name
                    ] += float(class_loss)

            else:
                loss = self.criterion(outputs, y)

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            seg_metrics.update(
                preds,
                y,
            )

            if verbose and batch_idx % 10 == 0:
                logger.debug(
                    f"Validation batch {batch_idx}/"
                    f"{len(val_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                }
            )

        if num_batches == 0:
            raise RuntimeError(
                "Validation dataloader is empty."
            )

        avg_loss = total_loss / num_batches
        metrics = seg_metrics.compute()

        if criterion_has_components:
            for key in (
                "total",
                "ce",
                "focal",
                "dice",
                "ce_focal_combined",
            ):
                metrics[f"val_loss_{key}"] = (
                    loss_components[key] / num_batches
                )

            per_class_total = cast(
                dict[str, float],
                loss_components["per_class"],
            )

            for class_name, class_loss in (
                per_class_total.items()
            ):
                metrics[f"val_loss_{class_name}"] = (
                    class_loss / num_batches
                )

        return avg_loss, metrics

    def _warmup_mps(self) -> None:
        """Warm up MPS shaders to avoid JIT compilation delay."""
        if self.device.type != "mps":
            return

        logger.info(
            "🔥 Warming up MPS shaders "
            "(first pass can take 2-10 minutes)..."
        )

        dummy_x = torch.randn(
            1,
            1,
            1578,
            751,
            device=self.device,
        )

        dummy_y = torch.randint(
            0,
            3,
            (1, 1578, 751),
            device=self.device,
        )

        dummy_out = self.model(dummy_x)
        dummy_loss = self.criterion(
            dummy_out,
            dummy_y,
        )

        if isinstance(dummy_loss, tuple):
            dummy_loss_tensor = cast(
                torch.Tensor,
                dummy_loss[0],
            )
        else:
            dummy_loss_tensor = cast(
                torch.Tensor,
                dummy_loss,
            )

        dummy_loss_tensor.backward()

        torch.mps.synchronize()

        self.model.zero_grad()
        torch.mps.empty_cache()

        logger.info("✅ MPS warmup complete!")

    def _log_model_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict | None = None,
    ) -> None:
        """Log a model checkpoint with MLflow registry support."""
        if (epoch + 1) % self.config.checkpoint_every != 0:
            return

        model_type = self.model_name
        dataset_name = self.config.dataset_name

        logger.info(
            "📦 _log_model_checkpoint: "
            f"checkpoint triggered at epoch {epoch + 1}"
        )
        logger.info(
            "   checkpoint_every: "
            f"{self.config.checkpoint_every}"
        )

        model_to_save = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )

        model_to_save.eval()

        sample_input = next(
            iter(self.dataloaders["val"])
        )[0][:1]

        registered_name = (
            f"{dataset_name.strip().replace(' ', '-')}-model"
        )

        logger.info(
            f"   Registered model name: {registered_name}"
        )
        logger.info(
            f"   Model type: {model_type}"
        )
        logger.info(
            f"   Dataset: {dataset_name}"
        )

        model_info: dict[str, Any] | None = None

        try:
            raw_model_info = (
                self.mlflow_manager.log_model_with_registry(
                    model=model_to_save,
                    model_name=(
                        f"{model_type}_"
                        f"{dataset_name}_"
                        f"epoch_{epoch + 1}"
                    ),
                    dataset_name=dataset_name,
                    step=epoch + 1,
                    registered_model_name=registered_name,
                    input_example=sample_input.cpu().numpy(),
                    tags={
                        "train_loss": str(train_loss),
                        "val_loss": str(val_loss),
                        "epoch": str(epoch + 1),
                        "model_type": model_type,
                    },
                )
            )

            model_info = cast(
                dict[str, Any],
                raw_model_info,
            )

            logger.info(
                "✅ Model checkpoint logged to MLflow: "
                f"{model_info.get('model_uri')}"
            )

        except Exception as e:  # noqa: BLE001
            logger.error(
                f"❌ Failed to log model checkpoint: {e}"
            )

        if model_info is not None:
            metrics: dict[str, float] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(
                    self.optimizer.param_groups[0]["lr"]
                ),
            }

            if val_metrics:
                metrics.update(
                    {
                        "val_iou": float(
                            val_metrics.get(
                                "mean_iou",
                                0.0,
                            )
                        ),
                        "val_f1": float(
                            val_metrics.get(
                                "mean_f1",
                                0.0,
                            )
                        ),
                        "val_accuracy": float(
                            val_metrics.get(
                                "accuracy",
                                0.0,
                            )
                        ),
                    }
                )

                iou_per_class = val_metrics.get(
                    "iou_per_class",
                    [],
                )

                for i, iou in enumerate(iou_per_class):
                    metrics[f"class_{i}_iou"] = float(iou)

            model_id = model_info.get("model_id")

            if model_id is not None:
                mlflow.log_metrics(
                    metrics,
                    step=epoch + 1,
                    model_id=model_id,
                )
            else:
                mlflow.log_metrics(
                    metrics,
                    step=epoch + 1,
                )

            if "registered_model_version" in model_info:
                self.registered_models[
                    registered_name
                ] = {
                    "version": model_info[
                        "registered_model_version"
                    ],
                    "val_loss": val_loss,
                }

            logger.info(
                "Model checkpoint logged to MLflow: "
                f"{model_info.get('model_uri')}"
            )

        else:
            logger.warning(
                "⚠️ MLflow model logging failed, "
                "skipping MLflow metrics"
            )

        self._save_checkpoint(
            epoch + 1,
            train_loss,
            val_loss,
        )

    def _update_model_aliases(
        self,
        best_val_loss: float,
    ) -> None:
        """
        Update model aliases based on performance.
        """
        del best_val_loss

        dataset_name = self.config.dataset_name

        registered_name = (
            f"{dataset_name.strip().replace(' ', '-')}-model"
        )

        if registered_name not in self.registered_models:
            return

        champion_info = (
            self.mlflow_manager.get_model_by_alias(
                registered_model_name=registered_name,
                alias="champion",
            )
        )

        current_version = self.registered_models[
            registered_name
        ]["version"]

        current_val_loss = float(
            self.registered_models[
                registered_name
            ]["val_loss"]
        )

        if champion_info:
            champion_metrics = (
                self.mlflow_manager.get_run_metrics(
                    champion_info.run_id
                )
            )

            champion_val_loss = float(
                champion_metrics.get(
                    "metrics",
                    {},
                ).get(
                    "val_loss",
                    float("inf"),
                )
            )

            if current_val_loss < champion_val_loss:
                self.mlflow_manager.set_model_alias(
                    registered_model_name=registered_name,
                    alias="champion",
                    version=current_version,
                )

                logger.info(
                    "🚀 New champion model! "
                    f"Version {current_version} "
                    f"with val_loss {current_val_loss:.4f}"
                )

                self.mlflow_manager.set_model_alias(
                    registered_model_name=registered_name,
                    alias="challenger",
                    version=champion_info.version,
                )

            else:
                self.mlflow_manager.set_model_alias(
                    registered_model_name=registered_name,
                    alias="challenger",
                    version=current_version,
                )

                logger.info(
                    "Challenger model "
                    f"version {current_version} "
                    f"with val_loss {current_val_loss:.4f}"
                )

        else:
            self.mlflow_manager.set_model_alias(
                registered_model_name=registered_name,
                alias="champion",
                version=current_version,
            )

            logger.info(
                "First champion model! "
                f"Version {current_version} "
                f"with val_loss {current_val_loss:.4f}"
            )

        self.mlflow_manager.set_model_alias(
            registered_model_name=registered_name,
            alias="staging",
            version=current_version,
        )

    def _log_sample_predictions(
        self,
        epoch: int,
    ) -> None:
        """Log sample predictions to TensorBoard and MLflow."""
        if (
            self.config.log_predictions_every <= 0
            or epoch % self.config.log_predictions_every != 0
        ):
            return

        logger.info(
            f"📸 Logging sample predictions "
            f"(epoch {epoch})..."
        )

        self.model.eval()

        num_samples = min(
            4,
            len(self.dataloaders["val"].dataset),
        )

        with torch.no_grad():
            for idx in range(num_samples):
                data, mask = (
                    self.dataloaders["val"]
                    .dataset[idx]
                )

                x = data.unsqueeze(0).to(
                    self.device
                )

                output = self.model(x)

                pred = (
                    torch.argmax(
                        output,
                        dim=1,
                    )
                    .cpu()
                    .numpy()[0]
                )

                data_np = data.numpy()[0]
                mask_np = mask.numpy()

                shot_id = (
                    self.dataloaders["val"]
                    .dataset
                    .get_shot_id(idx)
                )

                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(15, 5),
                )

                axes[0].imshow(
                    data_np.T,
                    cmap="seismic",
                    aspect="auto",
                    vmin=-np.percentile(
                        np.abs(data_np),
                        95,
                    ),
                    vmax=np.percentile(
                        np.abs(data_np),
                        95,
                    ),
                )

                axes[0].set_title(
                    f"Seismogram (Shot {shot_id})"
                )
                axes[0].set_xlabel("Trace")
                axes[0].set_ylabel("Sample")

                axes[1].imshow(
                    mask_np.T,
                    cmap="tab10",
                    aspect="auto",
                    vmin=0,
                    vmax=2,
                )

                axes[1].set_title("Ground Truth")
                axes[1].set_xlabel("Trace")
                axes[1].set_ylabel("Sample")

                axes[2].imshow(
                    pred.T,
                    cmap="tab10",
                    aspect="auto",
                    vmin=0,
                    vmax=2,
                )

                axes[2].set_title("Prediction")
                axes[2].set_xlabel("Trace")
                axes[2].set_ylabel("Sample")

                plt.tight_layout()

                self.writer.add_figure(
                    f"Seismogram/Shot_{shot_id}",
                    fig,
                    epoch,
                )

                temp_path = (
                    f"temp_prediction_"
                    f"{shot_id}_{epoch}.png"
                )

                try:
                    fig.savefig(
                        temp_path,
                        dpi=150,
                        bbox_inches="tight",
                    )

                    mlflow.log_artifact(
                        temp_path,
                        artifact_path=(
                            f"predictions/epoch_{epoch}"
                        ),
                    )

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                plt.close(fig)

    def fit(
        self,
        resume_from: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Main training loop with integrated logging."""
        start_epoch = 0

        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(
                resume_from
            )

        config_dict = self.config.to_dict()
        config_dict["model_name"] = self.model_name

        self.mlflow_manager.start_run(
            config_dict=config_dict,
            tags={
                "dataset": self.config.dataset_name,
                "model_type": self.model_name,
                "device": str(self.device),
                "experiment_type": "training",
            },
        )

        run_id = self.mlflow_manager.run_id

        logger.info(
            f"Starting training from epoch {start_epoch}"
        )
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"TensorBoard: {self.tb_dir}")

        if start_epoch == 0:
            sample_x = (
                next(
                    iter(
                        self.dataloaders["train"]
                    )
                )[0][:1]
                .to(self.device)
            )

            self.writer.add_graph(
                self.model,
                sample_x,
            )

        self._warmup_mps()

        best_val_loss = float("inf")
        best_val_iou = 0.0
        patience_counter = 0

        epoch_times: list[float] = []

        for epoch in range(
            start_epoch,
            self.config.n_epochs,
        ):
            epoch_start = datetime.now(
                timezone.utc
            )

            # --------------------------------------------------
            # Training
            # --------------------------------------------------
            train_loss, train_metrics = (
                self.train_epoch(
                    verbose=verbose
                )
            )

            logger.info(
                f"Epoch {epoch + 1}/"
                f"{self.config.n_epochs} - "
                f"Train Loss: {train_loss:.4f}"
            )

            logger.info(
                f"  Train IoU: "
                f"{train_metrics['mean_iou']:.4f}, "
                f"Train Acc: "
                f"{train_metrics['accuracy']:.4f}"
            )

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------
            val_loss, val_metrics = self.validate(
                verbose=verbose
            )

            logger.info(
                f"Epoch {epoch + 1}/"
                f"{self.config.n_epochs} - "
                f"Val Loss: {val_loss:.4f}"
            )

            logger.info(
                f"  Val IoU: "
                f"{val_metrics['mean_iou']:.4f}, "
                f"Val Acc: "
                f"{val_metrics['accuracy']:.4f}"
            )

            # --------------------------------------------------
            # Scheduler
            # --------------------------------------------------
            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.scheduler.step(val_loss)

            elif self.scheduler is not None:
                self.scheduler.step()

            current_lr = float(
                self.optimizer.param_groups[0]["lr"]
            )

            # --------------------------------------------------
            # TensorBoard logging
            # --------------------------------------------------
            self.writer.add_scalar(
                "Loss/train",
                train_loss,
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/train_iou",
                train_metrics["mean_iou"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/train_f1",
                train_metrics["mean_f1"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/train_accuracy",
                train_metrics["accuracy"],
                epoch,
            )

            if "loss_ce" in train_metrics:
                self.writer.add_scalar(
                    "Loss/components/train_ce",
                    train_metrics["loss_ce"],
                    epoch,
                )

                self.writer.add_scalar(
                    "Loss/components/train_focal",
                    train_metrics["loss_focal"],
                    epoch,
                )

                self.writer.add_scalar(
                    "Loss/components/train_dice",
                    train_metrics["loss_dice"],
                    epoch,
                )

                self.writer.add_scalar(
                    "Loss/components/train_total",
                    train_metrics["loss_total"],
                    epoch,
                )

                for key, value in train_metrics.items():
                    if key.startswith("loss_class_"):
                        self.writer.add_scalar(
                            f"Loss/per_class/train_{key}",
                            value,
                            epoch,
                        )

                if "loss_class_2" in train_metrics:
                    self.writer.add_scalar(
                        "Loss/per_class/train_strip",
                        train_metrics["loss_class_2"],
                        epoch,
                    )

            self.writer.add_scalar(
                "Loss/val",
                val_loss,
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/val_iou",
                val_metrics["mean_iou"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/val_f1",
                val_metrics["mean_f1"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/val_accuracy",
                val_metrics["accuracy"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/lr",
                current_lr,
                epoch,
            )

            if "val_loss_ce" in val_metrics:
                self.writer.add_scalar(
                    "Loss/components/val_ce",
                    val_metrics["val_loss_ce"],
                    epoch,
                )

                self.writer.add_scalar(
                    "Loss/components/val_focal",
                    val_metrics["val_loss_focal"],
                    epoch,
                )

                self.writer.add_scalar(
                    "Loss/components/val_dice",
                    val_metrics["val_loss_dice"],
                    epoch,
                )

                self.writer.add_scalar(
                    "Loss/components/val_total",
                    val_metrics["val_loss_total"],
                    epoch,
                )

                for key, value in val_metrics.items():
                    if key.startswith("val_loss_class_"):
                        self.writer.add_scalar(
                            f"Loss/per_class/val_{key}",
                            value,
                            epoch,
                        )

                if "val_loss_class_2" in val_metrics:
                    self.writer.add_scalar(
                        "Loss/per_class/val_strip",
                        val_metrics["val_loss_class_2"],
                        epoch,
                    )

            for i, iou in enumerate(
                val_metrics["iou_per_class"]
            ):
                self.writer.add_scalar(
                    f"Metrics/class_{i}_iou",
                    iou,
                    epoch,
                )

            # --------------------------------------------------
            # Memory logging
            # --------------------------------------------------
            memory: dict[str, float] = {}

            if self.config.log_memory:
                memory = self.get_memory_usage()

                if memory:
                    self.writer.add_scalar(
                        "Memory/allocated_GB",
                        memory["allocated"],
                        epoch,
                    )

                    self.writer.add_scalar(
                        "Memory/max_GB",
                        memory.get(
                            "max_allocated",
                            memory["allocated"],
                        ),
                        epoch,
                    )

                    logger.info(
                        "📊 Memory: "
                        f"allocated="
                        f"{memory['allocated']:.2f} GB"
                    )

            # --------------------------------------------------
            # Gradient / Weight norms
            # --------------------------------------------------
            if epoch % 5 == 0:
                grad_norm = compute_gradient_norm(
                    self.model
                )

                weight_norm = compute_weight_norm(
                    self.model
                )

                self.writer.add_scalar(
                    "Norms/gradient",
                    grad_norm,
                    epoch,
                )

                self.writer.add_scalar(
                    "Norms/weights",
                    weight_norm,
                    epoch,
                )

            # --------------------------------------------------
            # MLflow metrics
            # --------------------------------------------------
            mlflow_metrics: dict[str, float] = {
                "train_iou": float(
                    train_metrics["mean_iou"]
                ),
                "val_iou": float(
                    val_metrics["mean_iou"]
                ),
                "train_f1": float(
                    train_metrics["mean_f1"]
                ),
                "val_f1": float(
                    val_metrics["mean_f1"]
                ),
                "train_accuracy": float(
                    train_metrics["accuracy"]
                ),
                "val_accuracy": float(
                    val_metrics["accuracy"]
                ),
            }

            for i, iou in enumerate(
                val_metrics["iou_per_class"]
            ):
                mlflow_metrics[
                    f"val_class_{i}_iou"
                ] = float(iou)

            if self.config.log_memory and memory:
                mlflow_metrics[
                    "memory_allocated_gb"
                ] = memory["allocated"]

                mlflow_metrics[
                    "memory_max_gb"
                ] = memory.get(
                    "max_allocated",
                    memory["allocated"],
                )

            if "loss_ce" in train_metrics:
                mlflow_metrics.update(
                    {
                        "train_total_loss": float(
                            train_metrics["loss_total"]
                        ),
                        "train_ce_loss": float(
                            train_metrics["loss_ce"]
                        ),
                        "train_focal_loss": float(
                            train_metrics["loss_focal"]
                        ),
                        "train_dice_loss": float(
                            train_metrics["loss_dice"]
                        ),
                    }
                )

                for key, value in train_metrics.items():
                    if key.startswith("loss_class_"):
                        mlflow_metrics[
                            f"train_{key}"
                        ] = float(value)

            if "val_loss_ce" in val_metrics:
                mlflow_metrics.update(
                    {
                        "val_total_loss": float(
                            val_metrics["val_loss_total"]
                        ),
                        "val_ce_loss": float(
                            val_metrics["val_loss_ce"]
                        ),
                        "val_focal_loss": float(
                            val_metrics["val_loss_focal"]
                        ),
                        "val_dice_loss": float(
                            val_metrics["val_loss_dice"]
                        ),
                    }
                )

                for key, value in val_metrics.items():
                    if key.startswith(
                        "val_loss_class_"
                    ):
                        mlflow_metrics[
                            f"val_{key}"
                        ] = float(value)

            self.mlflow_manager.log_metrics(
                mlflow_metrics,
                step=epoch,
            )

            # --------------------------------------------------
            # Epoch timing
            # --------------------------------------------------
            epoch_duration = (
                datetime.now(timezone.utc)
                - epoch_start
            ).total_seconds()

            epoch_times.append(
                epoch_duration
            )

            self.mlflow_manager.log_metrics(
                {
                    "epoch_duration_seconds": (
                        epoch_duration
                    )
                },
                step=epoch,
            )

            logger.info(
                f"⏱ Epoch duration: "
                f"{epoch_duration:.1f}s"
            )

            # --------------------------------------------------
            # Model checkpoint
            # --------------------------------------------------
            self._log_model_checkpoint(
                epoch,
                train_loss,
                val_loss,
                val_metrics,
            )

            # --------------------------------------------------
            # Early stopping
            # --------------------------------------------------
            if (
                self.config.early_stopping_patience
                is not None
            ):
                current_val_iou = float(
                    val_metrics["mean_iou"]
                )

                if (
                    current_val_iou
                    > best_val_iou
                    + self.config.early_stopping_min_delta
                ):
                    best_val_iou = current_val_iou
                    best_val_loss = val_loss
                    patience_counter = 0

                    logger.info(
                        f"New best val IoU: "
                        f"{best_val_iou:.4f}"
                    )

                    self._save_best_model(
                        best_val_loss
                    )

                    self._update_model_aliases(
                        best_val_loss
                    )

                else:
                    patience_counter += 1

                    if (
                        patience_counter
                        >= self.config.early_stopping_patience
                    ):
                        logger.info(
                            "Early stopping triggered "
                            f"at epoch {epoch + 1}"
                        )
                        break

            # --------------------------------------------------
            # Local checkpoint
            # --------------------------------------------------
            if (
                (epoch + 1)
                % self.config.checkpoint_every
                == 0
            ):
                self._save_checkpoint(
                    epoch + 1,
                    train_loss,
                    val_loss,
                )

            # --------------------------------------------------
            # Sample predictions
            # --------------------------------------------------
            self._log_sample_predictions(epoch)

        self._update_model_aliases(
            best_val_loss
        )

        total_time = sum(epoch_times)

        self.mlflow_manager.log_metrics(
            {
                "total_training_time_seconds": (
                    total_time
                ),
                "average_epoch_time_seconds": (
                    total_time / len(epoch_times)
                    if epoch_times
                    else 0.0
                ),
            },
            step=epoch if "epoch" in locals() else 0,
        )

        self.writer.close()
        self.mlflow_manager.end_run()

        logger.info(
            "Training completed! "
            f"Best val_loss: {best_val_loss:.4f}"
        )

        logger.info(
            f"Model registry: {self.registry_dir}"
        )

        logger.info(
            f"TensorBoard: {self.tb_dir}"
        )

        logger.info(
            f"MLflow Run: {run_id}"
        )

        logger.info("=" * 60)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("=" * 60)

        logger.info(
            f"Dataset: {self.config.dataset_name}"
        )

        logger.info(
            f"Model: {self.model_name}"
        )

        logger.info(
            f"Epochs: {len(epoch_times)}"
        )

        logger.info(
            f"Best val_loss: {best_val_loss:.4f}"
        )

        logger.info(
            f"Best val IoU: {best_val_iou:.4f}"
        )

        if self.config.log_memory:
            logger.info(
                "Peak memory: "
                f"{self.get_memory_usage().get('max_allocated', 0):.2f} GB"
            )

        logger.info(
            f"Total training time: {total_time:.1f}s"
        )

        logger.info("=" * 60)

    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        """Save checkpoint to model registry."""
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        filename = (
            f"{self.model_name}_"
            f"{self.config.dataset_name}_"
            f"epoch_{epoch}_"
            f"{timestamp}.pt"
        )

        save_path = (
            self.registry_dir / filename
        )

        model_to_save = (
            self.model.module
            if isinstance(
                self.model,
                nn.DataParallel,
            )
            else self.model
        )

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": (
                model_to_save.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler
                else None
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            "dataset_name": (
                self.config.dataset_name
            ),
            "timestamp": timestamp,
        }

        torch.save(
            checkpoint,
            save_path,
        )

        logger.info(
            f"Checkpoint saved: {save_path}"
        )

        self.mlflow_manager.log_artifact(
            str(save_path),
            artifact_path="checkpoints",
        )

    def _save_best_model(
        self,
        best_val_loss: float,
    ) -> None:
        """Save the best model to registry."""
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        filename = (
            f"{self.model_name}_"
            f"{self.config.dataset_name}_"
            f"best_{timestamp}.pt"
        )

        save_path = (
            self.registry_dir / filename
        )

        model_to_save = (
            self.model.module
            if isinstance(
                self.model,
                nn.DataParallel,
            )
            else self.model
        )

        best_checkpoint: dict[str, Any] = {
            "model_state_dict": (
                model_to_save.state_dict()
            ),
            "val_loss": best_val_loss,
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            "dataset_name": (
                self.config.dataset_name
            ),
            "timestamp": timestamp,
        }

        torch.save(
            best_checkpoint,
            save_path,
        )

        logger.info(
            f"Best model saved: {save_path} "
            f"(val_loss: {best_val_loss:.4f})"
        )

        best_path = (
            self.registry_dir
            / (
                f"{self.model_name}_"
                f"{self.config.dataset_name}_"
                "best.pt"
            )
        )

        torch.save(
            best_checkpoint,
            best_path,
        )

        self.mlflow_manager.log_artifact(
            str(best_path),
            artifact_path="checkpoints",
        )
