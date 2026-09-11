"""
Unit tests for :mod:`scripts.search_models`.

Covers:
    * Filtering by dataset — single and multiple datasets.
    * Filtering by IoU threshold — ``--min-iou``.
    * Filtering by model type.
    * Sorting by metrics — ``val_iou`` descending.
    * Combined filters — dataset + model type + IoU.
    * CLI integration — filters forwarded to the MLflow manager.

The MLflow manager is mocked; no tracking server is contacted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import search_models

# ============================================================
# FAKE MLFLOW OBJECTS
# ============================================================


class _FakeMetric:
    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value


class _FakeTag:
    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value


class _FakeModel:
    """Minimal stand-in for a model returned by MLflow."""

    def __init__(
        self,
        name: str,
        dataset: str,
        model_type: str,
        val_iou: float,
        val_f1: float = 0.0,
        val_accuracy: float = 0.0,
        class_2_iou: float = 0.0,
        model_id: str = "m-000",
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.metrics = [
            _FakeMetric("val_iou", val_iou),
            _FakeMetric("val_f1", val_f1),
            _FakeMetric("val_accuracy", val_accuracy),
            _FakeMetric("class_2_iou", class_2_iou),
        ]
        self.tags = [
            _FakeTag("dataset", dataset),
            _FakeTag("model_type", model_type),
        ]


def _fake_models() -> list[_FakeModel]:
    """Deterministic list of fake models across datasets."""
    return [
        _FakeModel(
            "Halfmile_pico_e1", "Halfmile", "PicoUNet", 0.35, 0.40, 0.70, 0.30, "m-1"
        ),
        _FakeModel(
            "Halfmile_nano_e1", "Halfmile", "NanoUNet", 0.42, 0.48, 0.75, 0.38, "m-2"
        ),
        _FakeModel(
            "Halfmile_unet_e1", "Halfmile", "UNet", 0.68, 0.72, 0.88, 0.65, "m-3"
        ),
        _FakeModel(
            "Brunswick_pico_e1", "Brunswick", "PicoUNet", 0.30, 0.35, 0.68, 0.25, "m-4"
        ),
        _FakeModel(
            "Brunswick_mpslight_e1",
            "Brunswick",
            "MPSLightUNet",
            0.55,
            0.60,
            0.80,
            0.50,
            "m-5",
        ),
        _FakeModel("Lalor_tiny_e1", "Lalor", "TinyUNet", 0.48, 0.52, 0.76, 0.44, "m-6"),
    ]


def _build_filter_string(
    dataset: str | None,
    model_type: str | None,
    min_iou: float | None,
) -> str | None:
    """Reproduce the filter-building logic from ``scripts/search_models.py``."""
    filters: list[str] = []
    if dataset:
        filters.append(f"tags.dataset = '{dataset}'")
    if model_type:
        filters.append(f"tags.model_type = '{model_type}'")
    if min_iou is not None:
        filters.append(f"metrics.val_iou > {min_iou}")
    return " AND ".join(filters) if filters else None


def _apply_filter_locally(
    models: list[_FakeModel],
    dataset: str | None,
    model_type: str | None,
    min_iou: float | None,
) -> list[_FakeModel]:
    """Apply the same filter semantics the real search would apply."""
    result: list[_FakeModel] = []
    for m in models:
        tags = {t.key: t.value for t in m.tags}
        metrics = {x.key: x.value for x in m.metrics}
        if dataset and tags.get("dataset") != dataset:
            continue
        if model_type and tags.get("model_type") != model_type:
            continue
        if min_iou is not None and metrics.get("val_iou", 0) <= min_iou:
            continue
        result.append(m)
    return result


def _sort_models(models: list[_FakeModel]) -> list[_FakeModel]:
    """Sort by val_iou descending (as the real script requests)."""
    return sorted(
        models,
        key=lambda m: next(x.value for x in m.metrics if x.key == "val_iou"),
        reverse=True,
    )


# ============================================================
# 1. FILTER BY DATASET
# ============================================================


class TestFilterByDataset:
    """Tests that models are filtered correctly by dataset."""

    def test_filter_halfmile(self) -> None:
        """Filtering by ``Halfmile`` returns only Halfmile models."""
        filtered = _apply_filter_locally(_fake_models(), "Halfmile", None, None)

        assert len(filtered) == 3
        for m in filtered:
            tags = {t.key: t.value for t in m.tags}
            assert tags["dataset"] == "Halfmile"

    def test_filter_brunswick(self) -> None:
        """Filtering by ``Brunswick`` returns only Brunswick models."""
        filtered = _apply_filter_locally(_fake_models(), "Brunswick", None, None)

        assert len(filtered) == 2
        for m in filtered:
            tags = {t.key: t.value for t in m.tags}
            assert tags["dataset"] == "Brunswick"

    def test_filter_lalor(self) -> None:
        """Filtering by ``Lalor`` returns only Lalor models."""
        filtered = _apply_filter_locally(_fake_models(), "Lalor", None, None)

        assert len(filtered) == 1
        assert filtered[0].name.startswith("Lalor_")

    def test_filter_unknown_dataset_returns_empty(self) -> None:
        """An unknown dataset must produce an empty result."""
        filtered = _apply_filter_locally(_fake_models(), "NoSuchDataset", None, None)
        assert filtered == []

    def test_filter_string_for_single_dataset(self) -> None:
        """The filter string must contain ``tags.dataset = '...'``."""
        assert (
            _build_filter_string("Halfmile", None, None) == "tags.dataset = 'Halfmile'"
        )

    def test_filter_multiple_datasets_by_merging(self) -> None:
        """Merging two single-dataset filters yields both datasets' models."""
        models = _fake_models()
        half = _apply_filter_locally(models, "Halfmile", None, None)
        bruns = _apply_filter_locally(models, "Brunswick", None, None)
        merged = half + bruns

        datasets = {next(t.value for t in m.tags if t.key == "dataset") for m in merged}
        assert datasets == {"Halfmile", "Brunswick"}
        assert len(merged) == 5


# ============================================================
# 2. FILTER BY IoU THRESHOLD
# ============================================================


class TestFilterByIoU:
    """Tests for ``--min-iou`` filtering."""

    def test_min_iou_zero_returns_all(self) -> None:
        """A min-iou of 0 returns every model with val_iou > 0."""
        filtered = _apply_filter_locally(_fake_models(), None, None, 0.0)
        assert len(filtered) == len(_fake_models())

    def test_min_iou_threshold_filters_low_models(self) -> None:
        """A min-iou of 0.5 keeps only models with val_iou > 0.5."""
        filtered = _apply_filter_locally(_fake_models(), None, None, 0.5)
        for m in filtered:
            metrics = {x.key: x.value for x in m.metrics}
            assert metrics["val_iou"] > 0.5

    def test_min_iou_strictly_greater_than(self) -> None:
        """Comparison is strictly ``>`` (a model at exactly 0.5 is excluded)."""
        models = _fake_models()
        models.append(
            _FakeModel("Exact", "Halfmile", "UNet", 0.5, 0.5, 0.5, 0.5, "m-999")
        )
        filtered = _apply_filter_locally(models, None, None, 0.5)
        assert all(m.model_id != "m-999" for m in filtered)

    def test_min_iou_high_threshold(self) -> None:
        """A min-iou of 0.6 keeps only the single top model."""
        filtered = _apply_filter_locally(_fake_models(), None, None, 0.6)
        assert len(filtered) == 1
        assert filtered[0].model_id == "m-3"

    def test_filter_string_for_iou(self) -> None:
        """The filter string must contain ``metrics.val_iou > N``."""
        assert _build_filter_string(None, None, 0.65) == "metrics.val_iou > 0.65"


# ============================================================
# 3. FILTER BY MODEL TYPE
# ============================================================


class TestFilterByModelType:
    """Tests for ``--model-type`` filtering."""

    def test_filter_by_unet(self) -> None:
        """Filtering by ``UNet`` returns only UNet models."""
        filtered = _apply_filter_locally(_fake_models(), None, "UNet", None)
        for m in filtered:
            tags = {t.key: t.value for t in m.tags}
            assert tags["model_type"] == "UNet"
        assert len(filtered) == 1

    def test_filter_unknown_model_type_empty(self) -> None:
        """An unknown model type must produce an empty result."""
        filtered = _apply_filter_locally(_fake_models(), None, "NoSuchModel", None)
        assert filtered == []

    def test_filter_string_for_model_type(self) -> None:
        """The filter string must contain ``tags.model_type = '...'``."""
        assert _build_filter_string(None, "UNet", None) == "tags.model_type = 'UNet'"


# ============================================================
# 4. COMBINED FILTERS
# ============================================================


class TestCombinedFilters:
    """Tests for dataset + model type + IoU composition (AND)."""

    def test_dataset_and_iou(self) -> None:
        """Filtering by dataset AND min-iou must satisfy both."""
        filtered = _apply_filter_locally(_fake_models(), "Halfmile", None, 0.5)
        assert len(filtered) == 1
        assert filtered[0].model_id == "m-3"

    def test_dataset_and_model_type(self) -> None:
        """Filtering by dataset AND model type must satisfy both."""
        filtered = _apply_filter_locally(_fake_models(), "Halfmile", "NanoUNet", None)
        assert len(filtered) == 1
        assert filtered[0].model_id == "m-2"

    def test_all_three_filters(self) -> None:
        """Filtering by dataset + model type + min-iou must satisfy all."""
        filtered = _apply_filter_locally(_fake_models(), "Halfmile", "UNet", 0.5)
        assert len(filtered) == 1
        assert filtered[0].model_id == "m-3"

    def test_combined_filter_string(self) -> None:
        """Combined filters must be joined with ``AND``."""
        assert _build_filter_string("Halfmile", "UNet", 0.5) == (
            "tags.dataset = 'Halfmile' AND "
            "tags.model_type = 'UNet' AND "
            "metrics.val_iou > 0.5"
        )

    def test_contradictory_filters_empty(self) -> None:
        """Contradictory filters return empty."""
        assert _apply_filter_locally(_fake_models(), "Lalor", "UNet", None) == []


# ============================================================
# 5. SORTING BY METRICS
# ============================================================


class TestSortByMetrics:
    """Tests that results are sorted by ``val_iou`` descending."""

    def test_sorted_descending(self) -> None:
        """After sorting, ``val_iou`` must be non-increasing."""
        sorted_models = _sort_models(_fake_models())
        values = [
            next(x.value for x in m.metrics if x.key == "val_iou")
            for m in sorted_models
        ]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_highest_first(self) -> None:
        """The top-sorted model must have the highest ``val_iou``."""
        sorted_models = _sort_models(_fake_models())
        top_iou = next(x.value for x in sorted_models[0].metrics if x.key == "val_iou")
        assert top_iou == pytest.approx(0.68)

    def test_lowest_last(self) -> None:
        """The last-sorted model must have the lowest ``val_iou``."""
        sorted_models = _sort_models(_fake_models())
        last_iou = next(
            x.value for x in sorted_models[-1].metrics if x.key == "val_iou"
        )
        assert last_iou == pytest.approx(0.30)

    def test_filter_then_sort(self) -> None:
        """Filtering by dataset and then sorting keeps descending order."""
        filtered = _apply_filter_locally(_fake_models(), "Halfmile", None, None)
        sorted_models = _sort_models(filtered)
        values = [
            next(x.value for x in m.metrics if x.key == "val_iou")
            for m in sorted_models
        ]
        assert values == sorted(values, reverse=True)
        assert len(sorted_models) == 3


# ============================================================
# 6. CLI INTEGRATION
# ============================================================


class TestSearchModelsIntegration:
    """Integration tests that exercise the CLI and inspect the forwarded args."""

    def _invoke(
        self,
        dataset: str | None = None,
        model_type: str | None = None,
        min_iou: float | None = None,
        top: int = 10,
    ) -> mock.MagicMock:
        from click.testing import CliRunner

        mlflow_manager = mock.MagicMock()
        mlflow_manager.search_models.return_value = _fake_models()

        with mock.patch.object(
            search_models, "get_mlflow_manager", return_value=mlflow_manager
        ):
            runner = CliRunner()
            args = ["--top", str(top)]
            if dataset:
                args += ["--dataset", dataset]
            if model_type:
                args += ["--model-type", model_type]
            if min_iou is not None:
                args += ["--min-iou", str(min_iou)]

            result = runner.invoke(search_models.main, args)

        assert result.exit_code == 0, result.output
        return mlflow_manager

    def test_filter_string_forwarded_single_dataset(self) -> None:
        """The CLI must forward ``tags.dataset = '...'`` to the manager."""
        mgr = self._invoke(dataset="Halfmile")
        kwargs = mgr.search_models.call_args.kwargs
        assert kwargs["filter_string"] == "tags.dataset = 'Halfmile'"

    def test_filter_string_forwarded_combined(self) -> None:
        """The CLI must join combined filters with ``AND``."""
        mgr = self._invoke(dataset="Halfmile", model_type="UNet", min_iou=0.5)
        kwargs = mgr.search_models.call_args.kwargs
        assert kwargs["filter_string"] == (
            "tags.dataset = 'Halfmile' AND "
            "tags.model_type = 'UNet' AND "
            "metrics.val_iou > 0.5"
        )

    def test_no_filter_when_no_args(self) -> None:
        """With no filters, ``filter_string`` must be ``None``."""
        mgr = self._invoke()
        kwargs = mgr.search_models.call_args.kwargs
        assert kwargs["filter_string"] is None

    def test_top_maps_to_max_results(self) -> None:
        """``--top N`` must map to ``max_results=N``."""
        mgr = self._invoke(top=3)
        kwargs = mgr.search_models.call_args.kwargs
        assert kwargs["max_results"] == 3

    def test_ordering_by_val_iou_descending(self) -> None:
        """The CLI must request ``val_iou`` descending."""
        mgr = self._invoke()
        kwargs = mgr.search_models.call_args.kwargs
        assert kwargs["order_by"] == [
            {"field_name": "metrics.val_iou", "ascending": False}
        ]
