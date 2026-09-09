from unittest.mock import MagicMock

import pytest

# Import the module to be tested
from src.utils.memory_utils import (
    clear_memory,
    get_memory_usage,
    is_memory_low,
    log_memory_stats,
)


@pytest.fixture
def mock_psutil_memory(mocker):
    """Reusable mock fixture for psutil.virtual_memory[cite: 1]."""
    mock_mem = MagicMock()
    mock_mem.total = 16 * 10**9  # 16 GB[cite: 1]
    mock_mem.available = 4 * 10**9  # 4 GB[cite: 1]
    mock_mem.used = 12 * 10**9  # 12 GB[cite: 1]
    mock_mem.percent = 75.0  # 75% usage[cite: 1]

    mocker.patch("src.utils.memory_utils.psutil.virtual_memory", return_value=mock_mem)
    return mock_mem


@pytest.fixture
def mock_torch_no_gpu(mocker):
    """Simulates an environment where both CUDA and MPS are unavailable[cite: 1]."""
    mocker.patch("src.utils.memory_utils.torch.cuda.is_available", return_value=False)
    mocker.patch(
        "src.utils.memory_utils.torch.backends.mps.is_available", return_value=False
    )


class TestClearMemory:
    def test_clear_memory_should_clear_cuda_and_run_gc_when_cuda_available(
        self, mocker
    ):
        # Arrange
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.is_available", return_value=True
        )
        mocker.patch(
            "src.utils.memory_utils.torch.backends.mps.is_available", return_value=False
        )
        mock_cuda_empty_cache = mocker.patch(
            "src.utils.memory_utils.torch.cuda.empty_cache"
        )
        mock_gc = mocker.patch("src.utils.memory_utils.gc.collect")

        # Act
        clear_memory()

        # Assert
        mock_cuda_empty_cache.assert_called_once()
        mock_gc.assert_called_once()

    def test_clear_memory_should_clear_mps_and_run_gc_when_mps_available(self, mocker):
        # Arrange
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.is_available", return_value=False
        )
        mocker.patch(
            "src.utils.memory_utils.torch.backends.mps.is_available", return_value=True
        )
        mock_mps_empty_cache = mocker.patch(
            "src.utils.memory_utils.torch.mps.empty_cache"
        )
        mock_gc = mocker.patch("src.utils.memory_utils.gc.collect")

        # Act
        clear_memory()

        # Assert
        mock_mps_empty_cache.assert_called_once()
        mock_gc.assert_called_once()

    def test_clear_memory_should_only_run_gc_when_no_gpu_available(
        self, mocker, mock_torch_no_gpu
    ):
        # Arrange
        mock_gc = mocker.patch("src.utils.memory_utils.gc.collect")
        mock_cuda_empty_cache = mocker.patch(
            "src.utils.memory_utils.torch.cuda.empty_cache"
        )
        mock_mps_empty_cache = mocker.patch(
            "src.utils.memory_utils.torch.mps.empty_cache"
        )

        # Act
        clear_memory()

        # Assert
        mock_gc.assert_called_once()
        mock_cuda_empty_cache.assert_not_called()
        mock_mps_empty_cache.assert_not_called()


class TestGetMemoryUsage:
    def test_get_memory_usage_should_return_system_stats_only_when_no_gpu(
        self, mock_psutil_memory, mock_torch_no_gpu
    ):
        # Arrange / Act
        usage = get_memory_usage()

        # Assert
        assert "system" in usage
        assert "cuda" not in usage
        assert "mps" not in usage
        assert usage["system"]["total_gb"] == 16.0
        assert usage["system"]["available_gb"] == 4.0
        assert usage["system"]["used_gb"] == 12.0
        assert usage["system"]["percent"] == 75.0

    def test_get_memory_usage_should_include_cuda_stats_when_cuda_available(
        self, mocker, mock_psutil_memory
    ):
        # Arrange
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.is_available", return_value=True
        )
        mocker.patch(
            "src.utils.memory_utils.torch.backends.mps.is_available", return_value=False
        )
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.memory_allocated", return_value=2 * 10**9
        )
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.memory_reserved", return_value=3 * 10**9
        )
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.max_memory_allocated",
            return_value=4 * 10**9,
        )

        # Act
        usage = get_memory_usage()

        # Assert
        assert "cuda" in usage
        assert usage["cuda"]["allocated_gb"] == 2.0
        assert usage["cuda"]["reserved_gb"] == 3.0
        assert usage["cuda"]["max_allocated_gb"] == 4.0

    def test_get_memory_usage_should_include_mps_stats_when_mps_available(
        self, mocker, mock_psutil_memory
    ):
        # Arrange
        mocker.patch(
            "src.utils.memory_utils.torch.cuda.is_available", return_value=False
        )
        mocker.patch(
            "src.utils.memory_utils.torch.backends.mps.is_available", return_value=True
        )
        mocker.patch(
            "src.utils.memory_utils.torch.mps.current_allocated_memory",
            return_value=1 * 10**9,
        )
        mocker.patch(
            "src.utils.memory_utils.torch.mps.driver_allocated_memory",
            return_value=1.5 * 10**9,
        )

        # Act
        usage = get_memory_usage()

        # Assert
        assert "mps" in usage
        assert usage["mps"]["allocated_gb"] == 1.0
        assert usage["mps"]["driver_gb"] == 1.5


class TestIsMemoryLow:
    @pytest.mark.parametrize(
        "available_bytes, threshold_gb, expected_result",
        [
            (
                1 * 10**9,
                2.0,
                True,
            ),  # 1GB available, threshold 2.0GB -> Low memory[cite: 1]
            (
                3 * 10**9,
                2.0,
                False,
            ),  # 3GB available, threshold 2.0GB -> Sufficient memory[cite: 1]
            (
                2 * 10**9,
                2.0,
                False,
            ),  # Exactly at threshold (Edge case) -> Sufficient memory[cite: 1]
            (0, 2.0, True),  # 0GB available (Edge case) -> Low memory[cite: 1]
        ],
    )
    def test_is_memory_low_should_return_expected_when_conditions_met(
        self, mocker, available_bytes, threshold_gb, expected_result
    ):
        # Arrange
        mock_mem = MagicMock()
        mock_mem.available = available_bytes
        mocker.patch(
            "src.utils.memory_utils.psutil.virtual_memory", return_value=mock_mem
        )

        # Act
        result = is_memory_low(threshold_gb)

        # Assert
        assert result is expected_result


class TestLogMemoryStats:
    def test_log_memory_stats_should_log_system_and_gpu_info_correctly(self, mocker):
        # Arrange
        mock_usage = {
            "system": {
                "total_gb": 16.0,
                "available_gb": 4.0,
                "used_gb": 12.0,
                "percent": 75.0,
            },
            "cuda": {
                "allocated_gb": 2.0,
                "reserved_gb": 3.0,
                "max_allocated_gb": 4.0,
            },
            "mps": {
                "allocated_gb": 1.0,
                "driver_gb": 1.5,
            },
        }
        mocker.patch("src.utils.memory_utils.get_memory_usage", return_value=mock_usage)
        mock_logger_info = mocker.patch("src.utils.memory_utils.logger.info")
        prefix = "[Train]"

        # Act
        log_memory_stats(prefix)

        # Assert
        assert mock_logger_info.call_count == 4
        mock_logger_info.assert_any_call(f"{prefix} Memory Stats:")
        mock_logger_info.assert_any_call("  System: 12.0GB / 16.0GB (75.0%)")
        mock_logger_info.assert_any_call("  CUDA: 2.00GB allocated, 3.00GB reserved")
        mock_logger_info.assert_any_call("  MPS: 1.00GB allocated, 1.50GB driver")
