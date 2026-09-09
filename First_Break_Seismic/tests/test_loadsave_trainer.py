import torch
import torch.nn as nn

from src.training.trainer import SeismicTrainer


def create_test_trainer(tmp_path):
    """Create a minimal trainer without running full training setup."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = object.__new__(SeismicTrainer)

    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = None
    trainer.device = torch.device("cpu")
    trainer.registry_dir = tmp_path
    trainer.model_name = "test_model"

    class FakeConfig:
        dataset_name = "test_dataset"

        def to_dict(self):
            return {
                "dataset_name": "test_dataset",
            }

    trainer.config = FakeConfig()

    class FakeMLflow:
        def log_artifact(self, *args, **kwargs):
            pass

    trainer.mlflow_manager = FakeMLflow()

    return trainer


# ============================================================
# 1. Checkpoint save and load
# ============================================================


def test_checkpoint_save_and_load(tmp_path):
    trainer = create_test_trainer(tmp_path)

    trainer._save_checkpoint(
        epoch=5,
        train_loss=0.5,
        val_loss=0.4,
    )

    checkpoints = list(tmp_path.glob("*.pt"))

    assert len(checkpoints) == 1

    checkpoint = torch.load(
        checkpoints[0],
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 5
    assert checkpoint["train_loss"] == 0.5
    assert checkpoint["val_loss"] == 0.4
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


# ============================================================
# 2. Resume training from checkpoint
# ============================================================


def test_resume_from_checkpoint(tmp_path):
    trainer = create_test_trainer(tmp_path)

    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "epoch": 7,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": None,
            "val_loss": 0.25,
        },
        checkpoint_path,
    )

    start_epoch = trainer.load_checkpoint(
        str(checkpoint_path)
    )

    # fit() uses this returned value as start_epoch.
    assert start_epoch == 7


# ============================================================
# 3. Early stopping logic
# ============================================================


def test_early_stopping():
    patience = 2
    best_val_iou = 0.0
    patience_counter = 0
    stopped = False

    val_ious = [
        0.50,
        0.40,
        0.30,
    ]

    for val_iou in val_ious:
        if val_iou > best_val_iou + 0.0:
            best_val_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                stopped = True
                break

    assert stopped is True
    assert patience_counter == 2


# ============================================================
# 4. Gradient clipping
# ============================================================


def test_gradient_clipping():
    model = nn.Linear(2, 1)

    x = torch.tensor(
        [[1000.0, 1000.0]]
    )

    y = torch.tensor(
        [[0.0]]
    )

    criterion = nn.MSELoss()

    output = model(x)
    loss = criterion(output, y)

    loss.backward()

    max_norm = 1.0

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    )

    total_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is not None:
            param_norm = parameter.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    assert total_norm <= max_norm + 1e-6