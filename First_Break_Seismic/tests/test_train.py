"""
Tests for scripts/train.py CLI argument parsing.
"""

from click.testing import CliRunner
import pytest

from scripts.train import main


@pytest.fixture
def parsed_args(monkeypatch):
    """Capture parsed CLI arguments without running training."""

    captured = {}

    def fake_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main, "callback", fake_main)

    return captured


def test_config_is_required():
    """--config must be provided."""

    result = CliRunner().invoke(main, [])

    assert result.exit_code != 0
    assert "--config" in result.output


def test_config_option(parsed_args):
    """--config should be parsed correctly."""

    result = CliRunner().invoke(
        main,
        ["--config", "config.yaml"],
    )

    assert result.exit_code == 0
    assert parsed_args["config"] == "config.yaml"


@pytest.mark.parametrize(
    "model",
    [
        "unet",
        "efficient",
        "mobile",
        "light",
        "nano",
        "mpslight",
        "tiny",
        "pico",
    ],
)
def test_model_choices(model, parsed_args):
    """All supported model choices should be accepted."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--model",
            model,
        ],
    )

    assert result.exit_code == 0
    assert parsed_args["model"] == model


def test_invalid_model():
    """Invalid model should be rejected."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--model",
            "invalid",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_epochs_option(parsed_args):
    """--epochs should be parsed as an integer."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--epochs",
            "25",
        ],
    )

    assert result.exit_code == 0
    assert parsed_args["epochs"] == 25


def test_epochs_short_option(parsed_args):
    """-e should work as an alias for --epochs."""

    result = CliRunner().invoke(
        main,
        [
            "-c",
            "config.yaml",
            "-e",
            "25",
        ],
    )

    assert result.exit_code == 0
    assert parsed_args["epochs"] == 25


@pytest.mark.parametrize(
    "loss",
    [
        "cross_entropy",
        "focal",
        "dice",
        "combo",
    ],
)
def test_loss_choices(loss, parsed_args):
    """All supported loss choices should be accepted."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--loss",
            loss,
        ],
    )

    assert result.exit_code == 0
    assert parsed_args["loss"] == loss


def test_invalid_loss():
    """Invalid loss should be rejected."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--loss",
            "invalid",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_verbose_flag(parsed_args):
    """--verbose should be parsed as True."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert parsed_args["verbose"] is True


def test_log_memory_flag(parsed_args):
    """--log-memory should be parsed as True."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--log-memory",
        ],
    )

    assert result.exit_code == 0
    assert parsed_args["log_memory"] is True


def test_multiple_options(parsed_args):
    """Multiple CLI options should be parsed together."""

    result = CliRunner().invoke(
        main,
        [
            "--config",
            "config.yaml",
            "--model",
            "tiny",
            "--epochs",
            "20",
            "--loss",
            "combo",
            "--verbose",
            "--log-memory",
        ],
    )

    assert result.exit_code == 0

    assert parsed_args["config"] == "config.yaml"
    assert parsed_args["model"] == "tiny"
    assert parsed_args["epochs"] == 20
    assert parsed_args["loss"] == "combo"
    assert parsed_args["verbose"] is True
    assert parsed_args["log_memory"] is True


def test_default_values(parsed_args):
    """Important Click defaults should be parsed correctly."""

    result = CliRunner().invoke(
        main,
        ["--config", "config.yaml"],
    )

    assert result.exit_code == 0

    assert parsed_args["model"] == "unet"
    assert parsed_args["loss"] == "cross_entropy"
    assert parsed_args["checkpoint_every"] == 5
    assert parsed_args["early_stopping"] == 5
    assert parsed_args["dice_weight"] == 0.5
    assert parsed_args["focal_gamma"] == 2.0


def test_help():
    """CLI help should contain the main options."""

    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0

    for option in [
        "--config",
        "--model",
        "--epochs",
        "--loss",
        "--verbose",
        "--log-memory",
    ]:
        assert option in result.output