"""
Unit tests for the centralized logging configuration manager.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alembic.migration import Iterator

# Adjust the import path according to your actual project structure
import src.utils.logger as logger_module
from src.utils.logger import (
    SeismicLogger,
    create_task_name,
    get_logger,
    setup_logger,
)


@pytest.fixture(autouse=True)
def _reset_global_logger() -> Iterator[None]:
    """
    Fixture to reset the global logger instance before and after each test.
    This ensures test isolation.
    """
    # Arrange
    logger_module._global_logger = None
    yield
    # Cleanup
    logger_module._global_logger = None


@pytest.fixture
def mock_loguru(mocker: MagicMock) -> MagicMock:
    """
    Mock the loguru logger to prevent actual file writes during tests.
    """
    return mocker.patch("src.utils.logger.logger")


@pytest.fixture
def dummy_config() -> SimpleNamespace:
    """
    Provide a dummy config object with a dataset_name attribute[cite: 2].
    """
    return SimpleNamespace(dataset_name="Halfmile")


class TestSeismicLogger:
    """Test suite for the SeismicLogger class."""

    def test_seismic_logger_should_create_directories_when_initialized(
        self, tmp_path: Path, mock_loguru: MagicMock
    ) -> None:
        """
        Test that date-based subdirectories are created appropriately[cite: 2].
        """
        # Arrange
        log_dir = str(tmp_path / "logs")
        task_name = "test_task"

        # Act
        logger_instance = SeismicLogger(
            log_dir=log_dir, task_name=task_name, create_latest_symlink=False
        )

        # Assert
        assert logger_instance.log_dir.exists()
        assert logger_instance.date_dir.exists()
        assert logger_instance.log_dir.name == "logs"

    def test_seismic_logger_should_fallback_to_info_when_level_is_invalid(
        self, tmp_path: Path, mock_loguru: MagicMock
    ) -> None:
        """
        Test fallback mechanism for invalid log levels[cite: 2].
        """
        # Arrange
        invalid_level = "SUPER_DEBUG"

        # Act
        logger_instance = SeismicLogger(
            log_dir=str(tmp_path), level=invalid_level, create_latest_symlink=False
        )

        # Assert
        assert logger_instance.level == "INFO"

    @pytest.mark.parametrize(
        "valid_level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_seismic_logger_should_accept_valid_levels_when_initialized(
        self, tmp_path: Path, mock_loguru: MagicMock, valid_level: str
    ) -> None:
        """
        Test that all valid log levels are correctly assigned[cite: 2].
        """
        # Arrange & Act
        logger_instance = SeismicLogger(
            log_dir=str(tmp_path), level=valid_level, create_latest_symlink=False
        )

        # Assert
        assert logger_instance.level == valid_level

    def test_seismic_logger_should_setup_handlers_when_initialized(
        self, tmp_path: Path, mock_loguru: MagicMock
    ) -> None:
        """
        Test that logger handlers (Console, Main, Error, Debug, JSON) are added[cite: 2].
        """
        # Arrange & Act
        SeismicLogger(log_dir=str(tmp_path), level="DEBUG", create_latest_symlink=False)

        # Assert
        mock_loguru.remove.assert_called_once()
        # 1 console handler + 4 file handlers (main, error, debug, json) = 5 calls[cite: 2]
        assert mock_loguru.add.call_count == 5

    def test_seismic_logger_get_log_paths_should_return_correct_dict(
        self, tmp_path: Path, mock_loguru: MagicMock
    ) -> None:
        """
        Test that get_log_paths returns the correct dictionary structure[cite: 2].
        """
        # Arrange
        logger_instance = SeismicLogger(
            log_dir=str(tmp_path), create_latest_symlink=False
        )

        # Act
        paths = logger_instance.get_log_paths()

        # Assert
        assert "main" in paths
        assert "errors" in paths
        assert "debug" in paths
        assert "json" in paths
        assert isinstance(paths["main"], Path)

    def test_seismic_logger_create_symlink_should_skip_when_not_main_process(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """
        Test that symlink creation is skipped in worker processes[cite: 2].
        """
        # Arrange
        # Patch the underlying module directly since the import is local to the method
        mock_process = mocker.patch("multiprocessing.current_process")
        mock_process.return_value.name = "Worker-1"
        mock_symlink = mocker.patch("os.symlink")

        # Act
        SeismicLogger(log_dir=str(tmp_path), create_latest_symlink=True)

        # Assert
        mock_symlink.assert_not_called()

    def test_seismic_logger_create_symlink_should_handle_exceptions_gracefully(
        self, tmp_path: Path, mocker: MagicMock, mock_loguru: MagicMock
    ) -> None:
        """
        Test that symlink creation errors are caught and logged as debug[cite: 2].
        """
        # Arrange
        # Patch the underlying module directly since the import is local to the method
        mock_process = mocker.patch("multiprocessing.current_process")
        mock_process.return_value.name = "MainProcess"
        mocker.patch("os.symlink", side_effect=OSError("Permission denied"))

        # Act
        SeismicLogger(log_dir=str(tmp_path), create_latest_symlink=True)

        # Assert
        mock_loguru.debug.assert_called_with(
            "Could not create symlink: Permission denied"
        )


class TestGlobalLoggerFunctions:
    """Test suite for module-level global logger functions."""

    def test_setup_logger_should_set_global_instance(
        self, tmp_path: Path, mock_loguru: MagicMock
    ) -> None:
        """
        Test that setup_logger correctly initializes _global_logger[cite: 2].
        """
        # Arrange
        assert logger_module._global_logger is None

        # Act
        logger_result = setup_logger(
            task_name="train", log_dir=str(tmp_path), create_latest_symlink=False
        )

        # Assert
        assert logger_module._global_logger is not None
        assert isinstance(logger_module._global_logger, SeismicLogger)
        assert logger_result == mock_loguru

    def test_get_logger_should_initialize_default_when_global_is_none(
        self, mocker: MagicMock, mock_loguru: MagicMock
    ) -> None:
        """
        Test that get_logger creates a default instance if one does not exist[cite: 2].
        """
        # Arrange
        assert logger_module._global_logger is None
        # Mock SeismicLogger to avoid creating real directories
        mock_seismic_class = mocker.patch("src.utils.logger.SeismicLogger")
        mock_seismic_instance = mock_seismic_class.return_value
        mock_seismic_instance.get_logger.return_value = mock_loguru

        # Act
        logger_result = get_logger()

        # Assert
        mock_seismic_class.assert_called_once_with(task_name="general", level="INFO")
        assert logger_module._global_logger == mock_seismic_instance
        assert logger_result == mock_loguru


class TestCreateTaskName:
    """Test suite for the create_task_name helper function."""

    def test_create_task_name_should_format_correctly_without_model(
        self, dummy_config: SimpleNamespace
    ) -> None:
        """
        Test task name generation with only config and task type[cite: 2].
        """
        # Arrange
        task_type = "preprocess"

        # Act
        result = create_task_name(config=dummy_config, task_type=task_type)

        # Assert
        assert result == "preprocess_Halfmile"

    def test_create_task_name_should_format_correctly_with_model(
        self, dummy_config: SimpleNamespace
    ) -> None:
        """
        Test task name generation including a model name[cite: 2].
        """
        # Arrange
        task_type = "training"
        model_name = "unet"

        # Act
        result = create_task_name(
            config=dummy_config, task_type=task_type, model_name=model_name
        )

        # Assert
        assert result == "training_Halfmile_unet"
