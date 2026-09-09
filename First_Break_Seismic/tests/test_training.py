from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import SeismicConfig
from src.training.trainer import SeismicTrainer


@pytest.fixture
def mock_config(tmp_path):
    """Test üçün mock konfiqurasiya obyektini yaradır."""
    config = MagicMock(spec=SeismicConfig)

    config.device = "cpu"
    config.multi_gpu = False
    config.gpu_ids = []

    config.model_registry_dir = str(tmp_path / "registry")
    config.tensorboard_log_dir = str(tmp_path / "tb")

    config.dataset_name = "test_dataset"
    config.mlflow_experiment_name = "test_experiment"

    config.gradient_clip_value = 1.0
    config.lr_scheduler = None

    return config


@pytest.fixture
def dummy_data():
    """Forward və backward üçün keçərli tensor məlumatları."""
    x = torch.randn(2, 1, 16, 16)
    y = torch.randint(0, 3, (2, 16, 16))

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
    )

    return {
        "train": dataloader,
        "val": dataloader,
    }


@pytest.fixture
def mock_trainer_setup(mock_config, dummy_data):
    """Trainer üçün lazımi mock asılılıqları yaradır."""
    model = nn.Conv2d(
        1,
        3,
        kernel_size=1,
    )

    # Criterion mock
    criterion = MagicMock(spec=nn.Module)

    loss_val = torch.tensor(
        0.5,
        requires_grad=True,
    )

    criterion.side_effect = (
        lambda out, y: loss_val
    )

    criterion.to.return_value = criterion

    # Source code-da:
    #
    # self._criterion_returns_components()
    #
    # daxilində:
    #
    # getattr(self.criterion, "return_components", False)
    #
    # istifadə olunur.
    #
    # Ona görə burada attribute-u əvvəlcədən yaradırıq.
    criterion.return_components = False

    # Optimizer mock
    optimizer = MagicMock(
        spec=torch.optim.Optimizer
    )

    # Source code-da get_mlflow_manager yoxdur.
    # Birbaşa MLflowManager istifadə olunur.
    with patch(
        "src.training.trainer.MLflowManager"
    ) as mock_mlflow_manager, patch(
        "src.training.trainer.SummaryWriter"
    ) as mock_summary_writer:

        # MLflowManager mock instance
        mlflow_instance = mock_mlflow_manager.return_value

        # Trainer-in constructor-da bunlar istifadə olunmasa da,
        # gələcək testlər üçün təhlükəsiz mock edirik.
        mlflow_instance.start_run.return_value = None
        mlflow_instance.end_run.return_value = None
        mlflow_instance.log_metrics.return_value = None

        # SummaryWriter mock instance
        writer_instance = mock_summary_writer.return_value
        writer_instance.add_scalar.return_value = None
        writer_instance.add_figure.return_value = None
        writer_instance.add_graph.return_value = None
        writer_instance.close.return_value = None

        trainer = SeismicTrainer(
            model=model,
            dataloaders=dummy_data,
            criterion=criterion,
            optimizer=optimizer,
            config=mock_config,
            model_name="test_model",
        )

        yield (
            trainer,
            model,
            criterion,
            optimizer,
        )


def test_train_epoch_forward_backward_pass(
    mock_trainer_setup,
):
    """Forward və Backward keçidlərinin doğru işləməsini test edir."""
    trainer, _model, _criterion, optimizer = (
        mock_trainer_setup
    )

    avg_loss, _metrics = trainer.train_epoch(
        verbose=False
    )

    assert isinstance(
        avg_loss,
        float,
    )

    assert avg_loss == pytest.approx(
        0.5
    )

    optimizer.zero_grad.assert_called()
    optimizer.step.assert_called()


def test_train_epoch_gradient_clipping(
    mock_trainer_setup,
):
    """Gradient clipping parametrinin tətbiq olunmasını test edir."""
    trainer, _, _, _ = mock_trainer_setup

    trainer.config.gradient_clip_value = 0.5

    with patch(
        "torch.nn.utils.clip_grad_norm_"
    ) as mock_clip:

        trainer.train_epoch(
            verbose=False
        )

        assert mock_clip.called

        assert mock_clip.call_args[0][1] == 0.5


def test_train_epoch_metric_computation(
    mock_trainer_setup,
):
    """Metriklərin hesablanmasını test edir."""
    trainer, _, _, _ = mock_trainer_setup

    with patch(
        "src.training.trainer.SegmentationMetrics"
    ) as mock_metrics_cls:

        mock_metrics_inst = MagicMock()

        mock_metrics_inst.compute.return_value = {
            "mean_iou": 0.85,
            "accuracy": 0.90,
            "mean_f1": 0.88,
            "iou_per_class": [
                0.8,
                0.9,
                0.85,
            ],
        }

        mock_metrics_cls.return_value = (
            mock_metrics_inst
        )

        _avg_loss, metrics = (
            trainer.train_epoch(
                verbose=False
            )
        )

        mock_metrics_inst.update.assert_called()
        mock_metrics_inst.compute.assert_called_once()

        assert metrics["mean_iou"] == pytest.approx(
            0.85
        )

        assert metrics["accuracy"] == pytest.approx(
            0.90
        )

        assert metrics["mean_f1"] == pytest.approx(
            0.88
        )


def test_train_epoch_loss_components(
    mock_trainer_setup,
):
    """Criterion return_components=True olduqda detallı loss hesablamasını test edir."""
    trainer, _, _, _ = mock_trainer_setup

    loss_tensor = torch.tensor(
        0.4,
        requires_grad=True,
    )

    components = {
        "total": 0.4,
        "ce": 0.1,
        "focal": 0.1,
        "dice": 0.2,
        "ce_focal_combined": 0.2,
        "per_class": {
            "class_0": 0.1,
            "class_1": 0.3,
        },
    }

    # Burada spec_set=nn.Module istifadə etmirik.
    # Çünki return_components nn.Module-un standart
    # attribute-u deyil.
    criterion = MagicMock()

    criterion.return_components = True

    criterion.side_effect = (
        lambda out, y: (
            loss_tensor,
            components,
        )
    )

    criterion.to.return_value = criterion

    trainer.criterion = criterion

    _avg_loss, metrics = (
        trainer.train_epoch(
            verbose=False
        )
    )

    assert metrics["loss_total"] == pytest.approx(
        0.4
    )

    assert metrics["loss_ce"] == pytest.approx(
        0.1
    )

    assert metrics["loss_focal"] == pytest.approx(
        0.1
    )

    assert metrics["loss_dice"] == pytest.approx(
        0.2
    )

    assert metrics[
        "loss_ce_focal_combined"
    ] == pytest.approx(0.2)

    assert metrics["loss_class_0"] == pytest.approx(
        0.1
    )

    assert metrics["loss_class_1"] == pytest.approx(
        0.3
    )


def test_train_epoch_verbose_logging(
    mock_trainer_setup,
):
    """Verbose=True olduqda logger.debug çağırılmasını test edir."""
    trainer, _, _, _ = mock_trainer_setup

    with patch(
        "src.training.trainer.logger"
    ) as mock_logger:

        trainer.train_epoch(
            verbose=True
        )

        mock_logger.debug.assert_called()
