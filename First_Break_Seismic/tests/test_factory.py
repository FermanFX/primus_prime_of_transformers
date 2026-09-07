import pytest
from torch import nn

from src.models.factory import (
    MODEL_REGISTRY,
    create_model,
    get_model_info,
    list_models,
)


class TestModelRegistry:
    def test_registry_is_not_empty(self):
        assert MODEL_REGISTRY

    def test_list_models_matches_registry(self):
        assert list_models() == list(MODEL_REGISTRY.keys())

    @pytest.mark.parametrize(
        "model_name",
        [
            "unet",
            "mpslight",
            "light",
            "nano",
            "ultranano",
            "tiny",
            "pico",
            "mobile",
            "efficient",
        ],
    )
    def test_expected_models_are_registered(self, model_name):
        assert model_name in MODEL_REGISTRY


class TestCreateModel:
    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_create_all_registered_models(self, model_name):
        model = create_model(model_name)

        assert isinstance(model, nn.Module)

    def test_create_model_with_custom_channels(self):
        model = create_model(
            "tiny",
            in_channels=2,
            out_channels=4,
        )

        assert isinstance(model, nn.Module)

    def test_create_model_passes_kwargs(self):
        model = create_model(
            "tiny",
            in_channels=2,
            out_channels=4,
        )

        # Basic sanity check: model was created successfully.
        assert isinstance(model, nn.Module)

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("does_not_exist")

    def test_unknown_model_error_contains_available_models(self):
        with pytest.raises(ValueError) as exc_info:
            create_model("does_not_exist")

        error_message = str(exc_info.value)

        assert "Unknown model: 'does_not_exist'" in error_message

        for model_name in MODEL_REGISTRY:
            assert model_name in error_message


class TestListModels:
    def test_list_models_returns_list(self):
        models = list_models()

        assert isinstance(models, list)

    def test_list_models_contains_only_strings(self):
        models = list_models()

        assert all(isinstance(name, str) for name in models)

    def test_list_models_has_no_duplicates(self):
        models = list_models()

        assert len(models) == len(set(models))


class TestGetModelInfo:
    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_get_model_info_for_all_models(self, model_name):
        info = get_model_info(model_name)

        assert isinstance(info, dict)
        assert info["name"] == model_name
        assert info["class"] == MODEL_REGISTRY[model_name].__name__
        assert info["params"] is None or isinstance(info["params"], int)

    def test_model_info_contains_expected_keys(self):
        info = get_model_info("tiny")

        assert set(info.keys()) == {"name", "class", "params"}

    def test_model_info_parameter_count_is_positive(self):
        info = get_model_info("tiny")

        assert info["params"] is not None
        assert info["params"] > 0

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_info("does_not_exist")
