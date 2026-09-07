import pandas as pd
import pytest

import AaltoAD.report as report


def test_plot_threshold_method_follows_metric():
    assert report._plot_threshold_method("oracle.f1") == "oracle"
    assert report._plot_threshold_method("oracle_expanded.f1") == "oracle"
    assert report._plot_threshold_method("pot.f1") == "pot"
    assert report._plot_threshold_method("calibration_loss") == "pot"


def test_plot_threshold_picks_method_threshold_and_rejects_invalid():
    result = {"pot": {"threshold": 0.5}, "oracle": {"threshold": 0.2}}
    assert report._plot_threshold(result, "calibration_loss") == 0.5
    assert report._plot_threshold(result, "oracle.f1") == 0.2
    assert report._plot_threshold({"pot": {"threshold": 0.0}, "oracle": {}}, "pot.f1") is None
    assert report._plot_threshold({}, "oracle.f1") is None


def test_plot_y_top_never_below_threshold_and_uses_percentile():
    assert report._plot_y_top(pd.Series([0.1, 0.2, 0.3])) == pytest.approx(1.15)
    assert report._plot_y_top(pd.Series([10.0] * 100)) == pytest.approx(11.5)


def test_top_models_ranks_by_metric_and_direction():
    by_model = {
        "A": [{"oracle": {"f1": 0.5}, "calibration_loss": 3.0}],
        "B": [{"oracle": {"f1": 0.9}, "calibration_loss": 1.0}],
        "C": [{"calibration_loss": 2.0}],
    }
    assert report._top_models(by_model, "oracle.f1", n=2) == ["B", "A"]
    assert report._top_models(by_model, "calibration_loss", n=3) == ["B", "C", "A"]
    assert report._top_models(by_model, "oracle.f1", n=3)[-1] == "C"
