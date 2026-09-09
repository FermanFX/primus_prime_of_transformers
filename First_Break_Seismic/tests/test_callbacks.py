import torch
import torch.nn as nn

from src.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LoggingCallback,
)


# ============================================================
# Callback Base Class
# ============================================================


class TestCallback:

    def test_callback_is_abstract(self):
        # Callback abstract class olduğu üçün
        # birbaşa instantiate edilə bilməməlidir.
        try:
            Callback()
            assert False, "Callback should be abstract"
        except TypeError:
            pass


# ============================================================
# EarlyStoppingCallback
# ============================================================


class TestEarlyStoppingCallback:

    def test_initial_state(self):
        callback = EarlyStoppingCallback()

        assert callback.patience == 5
        assert callback.min_delta == 1e-4
        assert callback.mode == "min"
        assert callback.verbose is True

        assert callback.best_value is None
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_first_epoch_sets_best_value(self):
        callback = EarlyStoppingCallback(
            patience=3,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        assert callback.best_value == 0.5
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_improvement_resets_counter(self):
        callback = EarlyStoppingCallback(
            patience=3,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.4},
        )

        assert callback.best_value == 0.4
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_no_improvement_increases_counter(self):
        callback = EarlyStoppingCallback(
            patience=3,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.6},
        )

        assert callback.best_value == 0.5
        assert callback.counter == 1
        assert callback.should_stop is False

    def test_early_stopping_triggers(self):
        callback = EarlyStoppingCallback(
            patience=2,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.6},
        )

        assert callback.counter == 1
        assert callback.should_stop is False

        callback.on_epoch_end(
            epoch=2,
            metrics={"val_loss": 0.7},
        )

        assert callback.counter == 2
        assert callback.should_stop is True

    def test_improvement_after_bad_epoch_resets_counter(self):
        callback = EarlyStoppingCallback(
            patience=3,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.6},
        )

        assert callback.counter == 1

        callback.on_epoch_end(
            epoch=2,
            metrics={"val_loss": 0.4},
        )

        assert callback.best_value == 0.4
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_min_delta(self):
        callback = EarlyStoppingCallback(
            patience=3,
            min_delta=0.1,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        # Improvement = 0.05
        # min_delta = 0.1
        # Therefore this is NOT considered improvement.
        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.45},
        )

        assert callback.best_value == 0.5
        assert callback.counter == 1

    def test_max_mode(self):
        callback = EarlyStoppingCallback(
            patience=3,
            mode="max",
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.6},
        )

        assert callback.best_value == 0.6
        assert callback.counter == 0

    def test_loss_used_when_val_loss_missing(self):
        callback = EarlyStoppingCallback(
            patience=3,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"loss": 1.0},
        )

        assert callback.best_value == 1.0

        callback.on_epoch_end(
            epoch=1,
            metrics={"loss": 0.8},
        )

        assert callback.best_value == 0.8

    def test_missing_loss_does_nothing(self):
        callback = EarlyStoppingCallback(
            patience=3,
            verbose=False,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"accuracy": 0.9},
        )

        assert callback.best_value is None
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_epoch_start_does_nothing(self):
        callback = EarlyStoppingCallback(
            verbose=False,
        )

        callback.on_epoch_start(epoch=0)

        assert callback.best_value is None
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_batch_methods_do_nothing(self):
        callback = EarlyStoppingCallback(
            verbose=False,
        )

        callback.on_batch_start(batch=0)
        callback.on_batch_end(
            batch=0,
            loss=0.5,
        )

        assert callback.best_value is None
        assert callback.counter == 0


# ============================================================
# ModelCheckpointCallback
# ============================================================


class TestModelCheckpointCallback:

    def create_model(self):
        return nn.Linear(2, 1)

    def create_optimizer(self, model):
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
        )

    def test_initial_state(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            verbose=False,
        )

        assert callback.save_dir == tmp_path
        assert callback.save_best is True
        assert callback.save_every == 5
        assert callback.mode == "min"
        assert callback.monitor == "val_loss"

        assert callback.best_value is None
        assert callback.best_path is None

        assert tmp_path.exists()

    def test_save_directory_is_created(self, tmp_path):
        save_dir = tmp_path / "checkpoints"

        assert not save_dir.exists()

        ModelCheckpointCallback(
            save_dir=save_dir,
            verbose=False,
        )

        assert save_dir.exists()
        assert save_dir.is_dir()

    def test_saves_checkpoint_every_n_epochs(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_every=2,
            save_best=False,
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        checkpoint_1 = tmp_path / "checkpoint_epoch_1.pt"

        assert not checkpoint_1.exists()

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.4},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        checkpoint_2 = tmp_path / "checkpoint_epoch_2.pt"

        assert checkpoint_2.exists()

    def test_checkpoint_content(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_every=1,
            save_best=False,
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        metrics = {
            "val_loss": 0.25,
            "accuracy": 0.9,
        }

        callback.on_epoch_end(
            epoch=2,
            metrics=metrics,
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        checkpoint_path = tmp_path / "checkpoint_epoch_3.pt"

        assert checkpoint_path.exists()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        assert checkpoint["epoch"] == 3
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["scheduler_state_dict"] is None
        assert checkpoint["val_loss"] == 0.25
        assert checkpoint["accuracy"] == 0.9

    def test_best_model_is_saved(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_best=True,
            save_every=10,
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        best_path = tmp_path / "best_model.pt"

        assert best_path.exists()
        assert callback.best_value == 0.5
        assert callback.best_path == best_path

    def test_best_model_updates_on_improvement(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_best=True,
            save_every=10,
            mode="min",
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        assert callback.best_value == 0.5

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.3},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        assert callback.best_value == 0.3
        assert callback.best_path == tmp_path / "best_model.pt"

    def test_best_model_does_not_update_without_improvement(
        self,
        tmp_path,
    ):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_best=True,
            save_every=10,
            mode="min",
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"val_loss": 0.7},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        assert callback.best_value == 0.5

    def test_best_model_max_mode(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_best=True,
            save_every=10,
            mode="max",
            monitor="accuracy",
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        callback.on_epoch_end(
            epoch=0,
            metrics={"accuracy": 0.80},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        callback.on_epoch_end(
            epoch=1,
            metrics={"accuracy": 0.90},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        assert callback.best_value == 0.90

        best_path = tmp_path / "best_model.pt"

        assert best_path.exists()

    def test_no_best_model_when_save_best_false(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_best=False,
            save_every=10,
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

        best_path = tmp_path / "best_model.pt"

        assert not best_path.exists()
        assert callback.best_value is None
        assert callback.best_path is None

    def test_checkpoint_with_scheduler(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            save_every=1,
            save_best=False,
            verbose=False,
        )

        model = self.create_model()
        optimizer = self.create_optimizer(model)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
        )

        callback.on_epoch_end(
            epoch=0,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        checkpoint_path = tmp_path / "checkpoint_epoch_1.pt"

        assert checkpoint_path.exists()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        assert checkpoint["scheduler_state_dict"] is not None

    def test_on_epoch_start_does_nothing(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            verbose=False,
        )

        callback.on_epoch_start(epoch=0)

        assert callback.best_value is None
        assert callback.best_path is None

    def test_batch_methods_do_nothing(self, tmp_path):
        callback = ModelCheckpointCallback(
            save_dir=tmp_path,
            verbose=False,
        )

        callback.on_batch_start(batch=0)

        callback.on_batch_end(
            batch=0,
            loss=0.5,
        )

        assert callback.best_value is None


# ============================================================
# LoggingCallback
# ============================================================


class TestLoggingCallback:

    def test_initial_state(self):
        callback = LoggingCallback(
            writer=None,
            mlflow_manager=None,
            log_every=10,
            verbose=False,
        )

        assert callback.writer is None
        assert callback.mlflow_manager is None
        assert callback.log_every == 10
        assert callback.verbose is False
        assert callback.batch_losses == []

    def test_epoch_start_resets_batch_losses(self):
        callback = LoggingCallback(
            verbose=False,
        )

        callback.batch_losses = [
            1.0,
            2.0,
            3.0,
        ]

        callback.on_epoch_start(epoch=1)

        assert callback.batch_losses == []

    def test_batch_end_stores_loss(self):
        callback = LoggingCallback(
            verbose=False,
        )

        callback.on_batch_end(
            batch=0,
            loss=1.5,
        )

        callback.on_batch_end(
            batch=1,
            loss=0.5,
        )

        assert callback.batch_losses == [
            1.5,
            0.5,
        ]

    def test_batch_losses_are_stored_in_order(self):
        callback = LoggingCallback(
            verbose=False,
        )

        losses = [
            1.0,
            0.8,
            0.6,
            0.4,
        ]

        for batch, loss in enumerate(losses):
            callback.on_batch_end(
                batch=batch,
                loss=loss,
            )

        assert callback.batch_losses == losses

    def test_tensorboard_writer(self):
        class FakeWriter:

            def __init__(self):
                self.calls = []

            def add_scalar(self, key, value, epoch):
                self.calls.append(
                    (key, value, epoch)
                )

        writer = FakeWriter()

        callback = LoggingCallback(
            writer=writer,
            verbose=False,
        )

        metrics = {
            "loss": 0.5,
            "accuracy": 0.9,
        }

        callback.on_epoch_end(
            epoch=2,
            metrics=metrics,
        )

        assert len(writer.calls) == 2

        assert ("loss", 0.5, 2) in writer.calls
        assert ("accuracy", 0.9, 2) in writer.calls

    def test_mlflow_manager(self):
        class FakeMLflowManager:

            def __init__(self):
                self.calls = []

            def log_metrics(self, metrics, step):
                self.calls.append(
                    (metrics, step)
                )

        manager = FakeMLflowManager()

        callback = LoggingCallback(
            mlflow_manager=manager,
            verbose=False,
        )

        metrics = {
            "loss": 0.5,
            "accuracy": 0.9,
        }

        callback.on_epoch_end(
            epoch=3,
            metrics=metrics,
        )

        assert len(manager.calls) == 1
        assert manager.calls[0][0] == metrics
        assert manager.calls[0][1] == 3

    def test_batch_start_does_nothing(self):
        callback = LoggingCallback(
            verbose=False,
        )

        callback.on_batch_start(batch=0)

        assert callback.batch_losses == []

    def test_epoch_end_without_writer_or_mlflow(self):
        callback = LoggingCallback(
            writer=None,
            mlflow_manager=None,
            verbose=False,
        )

        metrics = {
            "loss": 0.5,
            "accuracy": 0.9,
        }

        # Should execute without error.
        callback.on_epoch_end(
            epoch=0,
            metrics=metrics,
        )

        assert callback.batch_losses == []

    def test_log_every_setting(self):
        callback = LoggingCallback(
            log_every=5,
            verbose=False,
        )

        assert callback.log_every == 5

        for batch in range(5):
            callback.on_batch_end(
                batch=batch,
                loss=1.0,
            )

        assert len(callback.batch_losses) == 5