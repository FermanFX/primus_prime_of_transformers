"""
Comprehensive unit test suite for scripts/check_device_memory.py targeting 90%+ code coverage.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest

from scripts.check_device_memory import (
    calculate_optimal_config,
    calculate_smart_config,
    check_memory_usage,
    generate_aggressive_variants_for_model,
    generate_all_variants,
    generate_nonlinear_variants,
    generate_smart_configurations,
    generate_smart_variants,
    get_actual_data_shape,
    get_cuda_memory_info,
    get_dataset_info,
    get_device_info,
    get_mps_memory_info,
    get_recommended_memory_limits,
)


@pytest.fixture
def mock_psutil_memory(mocker: MagicMock) -> MagicMock:
    """Fixture providing a mock virtual memory object from psutil."""
    mock_mem = MagicMock()
    mock_mem.total = 16 * (1024**3)
    mock_mem.available = 8 * (1024**3)
    mock_mem.used = 8 * (1024**3)
    mock_mem.percent = 50.0
    mock_mem.free = 4 * (1024**3)

    mocker.patch(
        "scripts.check_device_memory.psutil.virtual_memory", return_value=mock_mem
    )
    return mock_mem


class TestMemoryDetectionExtended:
    """Test suite for extended device info and memory checks."""

    def test_get_cuda_memory_info_should_handle_exception_gracefully(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.torch.cuda.is_available", return_value=True
        )
        mocker.patch(
            "scripts.check_device_memory.torch.cuda.device_count",
            side_effect=Exception("CUDA error"),
        )
        mock_logger_warning = mocker.patch("scripts.check_device_memory.logger.warning")

        # Act
        result = get_cuda_memory_info()

        # Assert
        assert result is None
        mock_logger_warning.assert_called_once()

    def test_get_mps_memory_info_should_handle_exception_gracefully(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.torch.backends.mps.is_available",
            return_value=True,
        )
        mocker.patch(
            "scripts.check_device_memory.psutil.virtual_memory",
            side_effect=Exception("RAM error"),
        )
        mock_logger_warning = mocker.patch("scripts.check_device_memory.logger.warning")

        # Act
        result = get_mps_memory_info()

        # Assert
        assert result is None
        mock_logger_warning.assert_called_once()

    def test_get_device_info_should_return_all_system_and_hardware_keys(
        self, mocker: MagicMock, mock_psutil_memory: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.torch.cuda.is_available", return_value=False
        )
        mocker.patch(
            "scripts.check_device_memory.torch.backends.mps.is_available",
            return_value=False,
        )

        # Act
        info = get_device_info()

        # Assert
        assert "system" in info
        assert "python" in info
        assert "pytorch" in info
        assert "cpu" in info
        assert info["cuda"] is None
        assert info["mps"] is None


class TestRecommendationsAndUsage:
    """Test suite for memory limit recommendations and usage telemetry."""

    def test_get_recommended_memory_limits_should_handle_all_device_types(self) -> None:
        # Arrange
        info = {
            "cpu": {"available_gb": 8.0},
            "cuda": {"devices": [{"total_gb": 10.0}]},
            "mps": {"recommended_limit_gb": 8.0, "max_safe_gb": 10.0},
        }

        recs = get_recommended_memory_limits(info)
        assert recs is not None

        cpu, cuda, mps = recs["cpu"], recs["cuda"], recs["mps"]

        assert cpu is not None
        assert cuda is not None
        assert mps is not None

        assert cpu["recommended_gb"] == 5.6
        assert cuda["recommended_gb"] == 8.0
        assert mps["recommended_gb"] == 8.0

    def test_get_recommended_memory_limits_should_handle_missing_devices(self) -> None:
        # Arrange
        info: dict = {}

        # Act
        recs = get_recommended_memory_limits(info)

        # Assert
        assert recs["cpu"] is None
        assert recs["cuda"] is None
        assert recs["mps"] is None

    def test_check_memory_usage_should_include_gpu_allocations_when_available(
        self, mocker: MagicMock, mock_psutil_memory: MagicMock
    ) -> None:
        # Arrange
        # check_memory_usage divides by 1e9 (decimal GB), so we set mock values using 1e9
        mock_psutil_memory.total = 16 * 1e9
        mock_psutil_memory.available = 8 * 1e9
        mock_psutil_memory.used = 8 * 1e9

        mocker.patch(
            "scripts.check_device_memory.torch.cuda.is_available", return_value=True
        )
        mocker.patch(
            "scripts.check_device_memory.torch.cuda.memory_allocated",
            return_value=10**9,
        )
        mocker.patch(
            "scripts.check_device_memory.torch.backends.mps.is_available",
            return_value=True,
        )

        mock_torch_mps = MagicMock()
        mock_torch_mps.current_allocated_memory.return_value = 2 * 10**9
        mocker.patch(
            "scripts.check_device_memory.torch.mps", mock_torch_mps, create=True
        )

        # Act
        usage = check_memory_usage()

        # Assert
        assert usage["total_gb"] == 16.0
        assert usage["gpu"]["cuda_allocated"] == 1.0
        assert usage["gpu"]["mps_allocated"] == 2.0


class TestVariantGenerators:
    """Test suite for progressive variant generators and non-linear options."""

    @pytest.mark.parametrize("device_type", ["mps", "cuda", "cpu"])
    def test_calculate_optimal_config_should_handle_different_devices(
        self, device_type: str
    ) -> None:
        # Act
        res = calculate_optimal_config("unet", 16.0, device_type)

        # Assert
        assert res["device_type"] == device_type
        assert "calculations" in res

    def test_generate_all_variants_should_return_dictionary_of_variants(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.get_all_model_recommendations",
            return_value={
                "pico": {
                    "params": 2000,
                    "optimal_batch_size": 4,
                    "optimal_cache_size": 2,
                    "recommended_memory_limit_gb": 2.0,
                }
            },
        )

        # Act
        result = generate_all_variants(8.0, "cuda")

        # Assert
        assert "pico" in result
        assert len(result["pico"]) > 0

    @pytest.mark.parametrize("model_name", ["pico", "unet"])
    def test_generate_aggressive_variants_for_model_should_return_variants(
        self, model_name: str
    ) -> None:
        # Act
        variants = generate_aggressive_variants_for_model(model_name, 16.0, "cuda")

        # Assert
        assert isinstance(variants, list)
        assert len(variants) == 5

    def test_generate_aggressive_variants_should_return_empty_on_unknown_model(
        self,
    ) -> None:
        # Act
        variants = generate_aggressive_variants_for_model("nonexistent", 16.0, "cuda")

        # Assert
        assert variants == []

    @pytest.mark.parametrize("model_name", ["pico", "mpslight", "light", "unet"])
    def test_generate_nonlinear_variants_should_cover_all_size_branches(
        self, model_name: str
    ) -> None:
        # Act
        variants = generate_nonlinear_variants(model_name, 16.0, "cuda")

        # Assert
        assert isinstance(variants, list)
        assert len(variants) > 0

    def test_generate_nonlinear_variants_should_return_empty_on_unknown_model(
        self,
    ) -> None:
        # Act
        variants = generate_nonlinear_variants("unknown", 16.0, "cuda")

        # Assert
        assert variants == []


class TestDatasetAndSmartConfig:
    """Test suite for dataset processing, actual data shape reading, and smart calculations."""

    def test_get_dataset_info_should_handle_large_and_small_shots(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch.object(Path, "exists", return_value=True)
        mock_data = {
            "config": {"target_traces": 100, "n_samples": 512, "chunk_size": 69},
            "total_shots": 250,
            "chunks": [{"file_size_mb": 10}],
        }
        mocker.patch("builtins.open", mock_open(read_data=json.dumps(mock_data)))

        # Act
        res = get_dataset_info("Halfmile")

        # Assert
        assert res["total_shots"] == 250
        assert res["num_chunks"] == 1

    def test_get_actual_data_shape_should_return_zeros_when_dir_missing(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch.object(Path, "exists", return_value=False)

        # Act
        shape = get_actual_data_shape("Halfmile")

        # Assert
        assert shape == (0, 0)

    def test_get_actual_data_shape_should_return_shape_when_file_loads(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch("pathlib.Path.glob", return_value=[Path("chunk_001.pt")])

        mock_chunk = {"data": MagicMock(shape=(10, 128, 512))}
        mocker.patch("torch.load", return_value=mock_chunk)

        # Act
        traces, samples = get_actual_data_shape("Halfmile")

        # Assert
        assert traces == 128
        assert samples == 512

    def test_get_actual_data_shape_should_handle_load_exception(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch("pathlib.Path.glob", return_value=[Path("chunk_001.pt")])
        mocker.patch("torch.load", side_effect=Exception("Corrupted file"))
        mock_logger_warning = mocker.patch("scripts.check_device_memory.logger.warning")

        # Act
        shape = get_actual_data_shape("Halfmile")

        # Assert
        assert shape == (0, 0)
        mock_logger_warning.assert_called_once()

    def test_calculate_smart_config_should_return_empty_on_unknown_model(self) -> None:
        # Act
        res = calculate_smart_config("invalid_model", "Halfmile", 16.0, "cuda")

        # Assert
        assert res == {}

    def test_calculate_smart_config_should_compute_successfully(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.get_dataset_info",
            return_value={"total_shots": 30, "num_chunks": 2},
        )

        # Act
        res = calculate_smart_config("pico", "Halfmile", 16.0, "cuda")

        # Assert
        assert res["model"] == "pico"
        assert "optimal_batch_size" in res

    def test_generate_smart_variants_should_return_empty_when_optimal_fails(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.calculate_smart_config", return_value={}
        )

        # Act
        variants = generate_smart_variants("pico", "Halfmile", 16.0, "cuda")

        # Assert
        assert variants == []

    def test_generate_smart_configurations_should_generate_for_all_models(
        self, mocker: MagicMock
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.calculate_smart_config",
            return_value={
                "model": "pico",
                "params": 2000,
                "optimal_batch_size": 4,
                "optimal_cache_size": 2,
                "recommended_memory_limit_gb": 2.0,
            },
        )

        # Act
        configs = generate_smart_configurations("Halfmile", 16.0, "cuda")

        # Assert
        assert "pico" in configs
        assert "unet" in configs


class TestMainExecutionFlow:
    """Test suite for the entrypoint main() execution loop across various devices."""

    @pytest.mark.parametrize(
        "cuda_avail, mps_avail",
        [
            (True, False),  # CUDA active path
            (False, True),  # MPS active path
            (False, False),  # CPU fallback path
        ],
    )
    def test_main_should_execute_and_save_config_file(
        self,
        mocker: MagicMock,
        mock_psutil_memory: MagicMock,
        cuda_avail: bool,
        mps_avail: bool,
    ) -> None:
        # Arrange
        mocker.patch(
            "scripts.check_device_memory.torch.cuda.is_available",
            return_value=cuda_avail,
        )
        mocker.patch(
            "scripts.check_device_memory.torch.backends.mps.is_available",
            return_value=mps_avail,
        )

        if cuda_avail:
            # ADD THESE MOCKS
            mocker.patch(
                "scripts.check_device_memory.torch.cuda.device_count",
                return_value=1,
            )
            mocker.patch(
                "scripts.check_device_memory.torch.cuda.memory_allocated",
                return_value=0,
            )
            mocker.patch(
                "scripts.check_device_memory.torch.cuda.memory_reserved",
                return_value=0,
            )

            mock_props = MagicMock()
            mock_props.name = "NVIDIA GPU"
            mock_props.total_memory = 8 * (1024**3)
            mocker.patch(
                "scripts.check_device_memory.torch.cuda.get_device_properties",
                return_value=mock_props,
            )
