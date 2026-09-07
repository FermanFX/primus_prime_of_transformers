import pytest
from src.config import SeismicConfig

class TestSeismicConfig:
    """Tests for SeismicConfig."""

    # ================================================================
    # Default configuration
    # ================================================================

    def test_default_config(self):
        """Default configuration should be created successfully."""
        config = SeismicConfig()

        assert config.dataset_name == "Halfmile"
        assert config.hdf5_path == "data/raw/Halfmile3D_add_geom_sorted.hdf5"
        assert config.chunk_dir == "data/chunks"

        assert config.target_traces == 1578
        assert config.n_samples == 751
        assert config.strip_width == 8
        assert config.chunk_size == 69

        assert config.batch_size == 4
        assert config.learning_rate == 1e-3
        assert config.n_epochs == 30
        assert config.device == "mps"

        assert config.class_weights == [0.2, 0.2, 0.6]

    # ================================================================
    # Custom configuration
    # ================================================================

    def test_custom_config(self):
        """Custom values should be stored correctly."""
        config = SeismicConfig(
            dataset_name="TestDataset",
            target_traces=100,
            n_samples=500,
            strip_width=10,
            chunk_size=20,
            batch_size=8,
            learning_rate=0.001,
            n_epochs=10,
            device="cpu",
        )

        assert config.dataset_name == "TestDataset"
        assert config.target_traces == 100
        assert config.n_samples == 500
        assert config.strip_width == 10
        assert config.chunk_size == 20
        assert config.batch_size == 8
        assert config.learning_rate == 0.001
        assert config.n_epochs == 10
        assert config.device == "cpu"

    # ================================================================
    # Data validation
    # ================================================================

    def test_target_traces_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="target_traces must be positive",
        ):
            SeismicConfig(target_traces=0)

        with pytest.raises(
            ValueError,
            match="target_traces must be positive",
        ):
            SeismicConfig(target_traces=-1)

    def test_n_samples_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="n_samples must be positive",
        ):
            SeismicConfig(n_samples=0)

        with pytest.raises(
            ValueError,
            match="n_samples must be positive",
        ):
            SeismicConfig(n_samples=-1)

    def test_strip_width_must_be_positive_and_even(self):
        with pytest.raises(
            ValueError,
            match="strip_width must be positive and even",
        ):
            SeismicConfig(strip_width=0)

        with pytest.raises(
            ValueError,
            match="strip_width must be positive and even",
        ):
            SeismicConfig(strip_width=3)

        with pytest.raises(
            ValueError,
            match="strip_width must be positive and even",
        ):
            SeismicConfig(strip_width=-2)

    def test_valid_strip_width(self):
        config = SeismicConfig(strip_width=10)

        assert config.strip_width == 10

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="chunk_size must be positive",
        ):
            SeismicConfig(chunk_size=0)

        with pytest.raises(
            ValueError,
            match="chunk_size must be positive",
        ):
            SeismicConfig(chunk_size=-1)

    def test_cache_size_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="cache_size must be positive",
        ):
            SeismicConfig(cache_size=0)

        with pytest.raises(
            ValueError,
            match="cache_size must be positive",
        ):
            SeismicConfig(cache_size=-1)

    # ================================================================
    # Split validation
    # ================================================================

    def test_splits_must_sum_to_one(self):
        with pytest.raises(
            ValueError,
            match=r"train\+val\+test split must equal 1.0",
        ):
            SeismicConfig(
                train_split=0.7,
                val_split=0.1,
                test_split=0.1,
            )

    def test_valid_splits(self):
        config = SeismicConfig(
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
        )

        assert config.train_split == 0.7
        assert config.val_split == 0.2
        assert config.test_split == 0.1

    # ================================================================
    # Training validation
    # ================================================================

    def test_batch_size_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="batch_size must be positive",
        ):
            SeismicConfig(batch_size=0)

        with pytest.raises(
            ValueError,
            match="batch_size must be positive",
        ):
            SeismicConfig(batch_size=-1)

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="learning_rate must be positive",
        ):
            SeismicConfig(learning_rate=0)

        with pytest.raises(
            ValueError,
            match="learning_rate must be positive",
        ):
            SeismicConfig(learning_rate=-0.001)

    def test_n_epochs_must_be_positive(self):
        with pytest.raises(
            ValueError,
            match="n_epochs must be positive",
        ):
            SeismicConfig(n_epochs=0)

        with pytest.raises(
            ValueError,
            match="n_epochs must be positive",
        ):
            SeismicConfig(n_epochs=-1)

    def test_num_workers_must_be_non_negative(self):
        with pytest.raises(
            ValueError,
            match="num_workers must be non-negative",
        ):
            SeismicConfig(num_workers=-1)

    # ================================================================
    # Class weights validation
    # ================================================================

    def test_class_weights_must_have_three_values(self):
        with pytest.raises(
            ValueError,
            match="class_weights must have exactly 3 values",
        ):
            SeismicConfig(class_weights=[0.5, 0.5])

        with pytest.raises(
            ValueError,
            match="class_weights must have exactly 3 values",
        ):
            SeismicConfig(class_weights=[0.2, 0.3, 0.5, 0.1])

    def test_class_weights_must_be_non_negative(self):
        with pytest.raises(
            ValueError,
            match="class_weights must be non-negative",
        ):
            SeismicConfig(class_weights=[0.2, -0.1, 0.9])

    def test_valid_class_weights(self):
        config = SeismicConfig(class_weights=[0.1, 0.2, 0.7])

        assert config.class_weights == [0.1, 0.2, 0.7]

    # ================================================================
    # Log level validation
    # ================================================================

    @pytest.mark.parametrize(
        "log_level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_valid_log_levels(self, log_level):
        config = SeismicConfig(log_level=log_level)

        assert config.log_level == log_level

    def test_log_level_lowercase_is_accepted(self):
        config = SeismicConfig(log_level="info")

        assert config.log_level == "info"

    def test_invalid_log_level(self):
        with pytest.raises(
            ValueError,
            match="log_level must be one of",
        ):
            SeismicConfig(log_level="INVALID")

    # ================================================================
    # Scheduler validation
    # ================================================================

    @pytest.mark.parametrize(
        "scheduler",
        ["step", "plateau", "cosine"],
    )
    def test_valid_schedulers(self, scheduler):
        config = SeismicConfig(lr_scheduler=scheduler)

        assert config.lr_scheduler == scheduler

    def test_invalid_scheduler(self):
        with pytest.raises(
            ValueError,
            match="lr_scheduler must be one of",
        ):
            SeismicConfig(lr_scheduler="invalid")

    # ================================================================
    # Device validation
    # ================================================================

    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda", "mps"],
    )
    def test_valid_devices(self, device):
        config = SeismicConfig(device=device)

        assert config.device == device

    def test_invalid_device(self):
        with pytest.raises(
            ValueError,
            match="device must be one of",
        ):
            SeismicConfig(device="tpu")

    # ================================================================
    # Config hash
    # ================================================================

    def test_config_hash_is_string(self):
        config = SeismicConfig()

        config_hash = config.get_config_hash()

        assert isinstance(config_hash, str)
        assert len(config_hash) == 8

    def test_same_config_has_same_hash(self):
        config1 = SeismicConfig()
        config2 = SeismicConfig()

        assert config1.get_config_hash() == config2.get_config_hash()

    def test_different_config_has_different_hash(self):
        config1 = SeismicConfig()
        config2 = SeismicConfig(batch_size=8)

        assert config1.get_config_hash() != config2.get_config_hash()

    def test_hash_changes_when_learning_rate_changes(self):
        config1 = SeismicConfig(learning_rate=0.001)
        config2 = SeismicConfig(learning_rate=0.0001)

        assert config1.get_config_hash() != config2.get_config_hash()

    def test_hash_changes_when_dataset_changes(self):
        config1 = SeismicConfig(dataset_name="Halfmile")
        config2 = SeismicConfig(dataset_name="Brunswick")

        assert config1.get_config_hash() != config2.get_config_hash()

    # ================================================================
    # to_dict
    # ================================================================

    def test_to_dict_returns_dictionary(self):
        config = SeismicConfig()

        result = config.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_contains_expected_values(self):
        config = SeismicConfig()

        result = config.to_dict()

        assert result["dataset_name"] == "Halfmile"
        assert result["target_traces"] == 1578
        assert result["n_samples"] == 751
        assert result["strip_width"] == 8
        assert result["batch_size"] == 4
        assert result["learning_rate"] == 1e-3
        assert result["device"] == "mps"

    def test_to_dict_contains_all_public_fields(self):
        config = SeismicConfig()

        result = config.to_dict()

        for key in config.__dict__:
            if not key.startswith("_"):
                assert key in result

    # ================================================================
    # to_yaml
    # ================================================================

    def test_to_yaml_returns_string(self):
        pytest.importorskip("yaml")

        config = SeismicConfig()

        result = config.to_yaml()

        assert isinstance(result, str)

    def test_to_yaml_contains_config_values(self):
        pytest.importorskip("yaml")

        config = SeismicConfig()

        result = config.to_yaml()

        assert "dataset_name: Halfmile" in result
        assert "target_traces: 1578" in result
        assert "n_samples: 751" in result
        assert "batch_size: 4" in result

    # ================================================================
    # repr
    # ================================================================

    def test_repr(self):
        config = SeismicConfig()

        result = repr(config)

        assert result.startswith("SeismicConfig(")
        assert "dataset_name=Halfmile" in result
        assert "target_traces=1578" in result
        assert "n_samples=751" in result
        assert "batch_size=4" in result

    # ================================================================
    # Dataclass default_factory
    # ================================================================

    def test_class_weights_are_independent_between_instances(self):
        """default_factory should create a new list per instance."""
        config1 = SeismicConfig()
        config2 = SeismicConfig()

        config1.class_weights[0] = 999

        assert config1.class_weights[0] == 999
        assert config2.class_weights[0] == 0.2

    # ================================================================
    # Optional values
    # ================================================================

    def test_optional_values_accept_none(self):
        config = SeismicConfig(
            gradient_clip_value=None,
            early_stopping_patience=None,
            log_batch_every=None,
            gpu_ids=None,
        )

        assert config.gradient_clip_value is None
        assert config.early_stopping_patience is None
        assert config.log_batch_every is None
        assert config.gpu_ids is None

    # ================================================================
    # GPU configuration
    # ================================================================

    def test_gpu_configuration(self):
        config = SeismicConfig(
            multi_gpu=True,
            gpu_ids=[0, 1],
        )

        assert config.multi_gpu is True
        assert config.gpu_ids == [0, 1]
