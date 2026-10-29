import json

import pandas as pd
import pytest

import AaltoAD.report.report as report
from AaltoAD.report import cli, metrics, plot, tables


def test_method_name_normalises_dotted_metrics():
    assert cli.method_name("conformal") == "conformal"
    assert cli.method_name("conformal.F1") == "conformal"
    assert cli.method_name("oracle_expanded.F1") == "oracle"
    assert cli.method_name("pot.threshold") == "pot"


def test_plot_threshold_picks_method_threshold_and_rejects_invalid():
    result = {"pot": {"threshold": 0.5}, "conformal": {"threshold": 0.2}}
    assert plot._plot_threshold(result, "pot") == 0.5
    assert plot._plot_threshold(result, "conformal") == 0.2
    assert plot._plot_threshold({"pot": {"threshold": 0.0}}, "pot") is None
    assert plot._plot_threshold({}, "conformal") is None


def test_clip_for_plot_caps_values_at_the_shared_scale():
    values = pd.Series([0.5, 1.0, 2.5, 100.0], index=[-1, 0, 1, 2])
    clipped = plot._clip_for_plot(values)
    assert clipped.max() == plot.PLOT_VALUE_CLIP
    assert clipped.tolist() == [0.5, 1.0, 2.0, 2.0]


def test_plot_rank_key_follows_the_given_order():
    slow_but_accurate = {"pot": {"P_latency": 50, "FPR": 0.0, "F1": 0.9}}
    fast_but_poor = {"pot": {"P_latency": 5, "FPR": 0.1, "F1": 0.1}}
    by_latency = ("P_latency", "FPR", "F1")
    by_f1 = ("F1",)
    assert metrics.rank_key(fast_but_poor, by_latency, "pot") < \
        metrics.rank_key(slow_but_accurate, by_latency, "pot")
    assert metrics.rank_key(slow_but_accurate, by_f1, "pot") < \
        metrics.rank_key(fast_but_poor, by_f1, "pot")


def test_plot_rank_key_prefers_latency_then_fpr_then_f1():
    fast = {"pot": {"P_latency": 5, "FPR": 0.1, "F1": 0.1}}
    slow = {"pot": {"P_latency": 50, "FPR": 0.0, "F1": 0.9}}
    assert metrics.rank_key(fast, ("P_latency", "FPR", "F1"), "pot") < metrics.rank_key(slow, ("P_latency", "FPR", "F1"), "pot")


    clean = {"pot": {"P_latency": 5, "FPR": 0.01, "F1": 0.1}}
    noisy = {"pot": {"P_latency": 5, "FPR": 0.2, "F1": 0.9}}
    assert metrics.rank_key(clean, ("P_latency", "FPR", "F1"), "pot") < metrics.rank_key(noisy, ("P_latency", "FPR", "F1"), "pot")

    good = {"pot": {"P_latency": 5, "FPR": 0.01, "F1": 0.9}}
    poor = {"pot": {"P_latency": 5, "FPR": 0.01, "F1": 0.2}}
    assert metrics.rank_key(good, ("P_latency", "FPR", "F1"), "pot") < metrics.rank_key(poor, ("P_latency", "FPR", "F1"), "pot")


def test_plot_rank_key_sorts_missing_values_last():
    present = {"pot": {"P_latency": 1000, "FPR": 0.9, "F1": 0.0}}
    missing = {"pot": {}}
    assert metrics.rank_key(present, ("P_latency", "FPR", "F1"), "pot") < metrics.rank_key(missing, ("P_latency", "FPR", "F1"), "pot")
    assert metrics.rank_key(None, ("P_latency", "FPR", "F1"), "pot") == metrics.rank_key(missing, ("P_latency", "FPR", "F1"), "pot")


def test_top_models_ranks_by_latency_then_fpr_then_f1():
    by_model = {
        # A detects slowest, B fastest; C never fires and has no metrics at all.
        "A": [{"conformal": {"P_latency": 40, "FPR": 0.0, "F1": 0.9}}],
        "B": [{"conformal": {"P_latency": 5, "FPR": 0.1, "F1": 0.5}}],
        "C": [{"calibration_loss": 2.0}],
    }
    assert report._top_models(by_model, "latency,fpr,f1", n=2) == ["B", "A"]
    assert report._top_models(by_model, "latency,fpr,f1", n=3)[-1] == "C"


def test_top_models_reads_the_threshold_method_it_is_given():
    by_model = {
        "A": [{"pot": {"P_latency": 90, "FPR": 0.0, "F1": 0.1},
               "conformal": {"P_latency": 1, "FPR": 0.0, "F1": 0.9}}],
        "B": [{"pot": {"P_latency": 2, "FPR": 0.0, "F1": 0.1},
               "conformal": {"P_latency": 80, "FPR": 0.0, "F1": 0.9}}],
    }
    assert report._top_models(by_model, "latency,fpr,f1", n=2, threshold_method="pot") == ["B", "A"]
    assert report._top_models(by_model, "latency,fpr,f1", n=2, threshold_method="conformal") == ["A", "B"]


def test_top_models_honours_a_custom_model_order():
    by_model = {
        "A": [{"conformal": {"P_latency": 40, "FPR": 0.0, "F1": 0.9}}],
        "B": [{"conformal": {"P_latency": 5, "FPR": 0.1, "F1": 0.5}}],
    }
    assert report._top_models(by_model, "latency,fpr,f1", n=2, model_order=("F1",)) == ["A", "B"]
    assert report._top_models(by_model, "latency,fpr,f1", n=2, model_order=("FPR",)) == ["A", "B"]
    assert report._top_models(by_model, "latency,fpr,f1", n=2, model_order=("P_latency",)) == ["B", "A"]


def test_order_rank_gives_positions_in_display_order():
    by_model = {
        "slow": [{"conformal": {"P_latency": 40, "FPR": 0.0, "F1": 0.9}}],
        "fast": [{"conformal": {"P_latency": 5, "FPR": 0.1, "F1": 0.5}}],
    }
    assert report._order_rank(by_model, "latency,fpr,f1") == {"fast": 0, "slow": 1}


def test_sort_rows_follows_the_order_when_given():
    rows = [(0.9, {"model": "slow"}), (0.5, {"model": "fast"})]
    tables._sort_rows(rows, {"fast": 0, "slow": 1}, lower_is_better=False)
    assert [r["model"] for _, r in rows] == ["fast", "slow"]
    # Without an order, rows fall back to their own metric value.
    tables._sort_rows(rows, None, lower_is_better=False)
    assert [r["model"] for _, r in rows] == ["slow", "fast"]


def test_metric_list_accepts_the_spellings_the_cli_allows():
    assert cli.metric_list("latency,fpr,f1") == ("P_latency", "FPR", "F1")
    assert cli.metric_list("conformal.F1") == ("F1",)
    assert cli.metric_list(("P_latency", "F1")) == ("P_latency", "F1")
    assert cli.metric_list("calibration_loss") == ("calibration_loss",)


def test_metric_path_keeps_top_level_metrics_out_of_the_method_block():
    assert metrics._metric_path("F1", "conformal") == "conformal.F1"
    assert metrics._metric_path("calibration_loss", "conformal") == "calibration_loss"


def test_latex_columns_default_to_the_paper_set():
    assert tables._latex_columns("conformal") == [
        ("F1", "conformal.F1"),
        ("FPR", "conformal.FPR"),
        ("Latency", "conformal.P_latency"),
    ]


def test_latex_columns_take_the_names_they_are_given():
    columns = tables._latex_columns("pot", "f1,adjusted_f1,eval_time")
    assert columns == [
        ("F1", "pot.F1"),
        ("Adjusted F1", "pot_expanded.F1"),
        ("Eval time (s)", "eval_time"),
    ]
    # An unlisted name is read from the method's own block.
    assert tables._latex_columns("pot", "recall")[0] == ("Recall", "pot.recall")


def test_latex_tabular_deals_rows_into_blocks_column_major():
    rows = [{"model": f"M{i}"} for i in range(5)]
    table = tables._latex_tabular(["model"], rows, blocks=2)
    assert r"\begin{tabular}{l@{\qquad}l}" in table
    assert r"model & model \\" in table
    # Five rows over two blocks: three lines, the last with an empty right cell.
    assert r"M0 & M3 \\" in table
    assert r"M2 &  \\" in table


def test_blocked_rows_keeps_order_within_a_block():
    rows = list(range(6))
    assert tables._blocked_rows(rows, 2) == [(0, 3), (1, 4), (2, 5)]
    assert tables._blocked_rows(rows, 1) == [(0,), (1,), (2,), (3,), (4,), (5,)]


def test_rank_key_ignores_the_latency_of_a_model_that_never_fired():
    # Both "detect" at step 100, but `never` has no true positives, so its
    # latency is just the segment length and must not win the comparison.
    detected = {"conformal": {"P_latency": 1790, "FPR": 0.0, "F1": 0.01, "TP": 5, "FP": 0, "FN": 10}}
    never = {"conformal": {"P_latency": 1734, "FPR": 0.0, "F1": 0.0, "TP": 0, "FP": 0, "FN": 1734}}
    order = ("P_latency", "FPR", "F1")
    assert metrics.rank_key(detected, order, "conformal") < metrics.rank_key(never, order, "conformal")


def _write_model_order(tmp_path, dataset, names):
    reports = tmp_path / "reports" / dataset
    reports.mkdir(parents=True)
    (reports / "model-order.json").write_text(json.dumps(names))


def test_plot_model_selection_reads_each_datasets_own_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_model_order(tmp_path, "ds1", ["B", "A"])
    _write_model_order(tmp_path, "ds2", ["A"])
    by_model = {"A": ["rA"], "B": ["rB"], "C": ["rC"]}
    # Exactly the named models, in the file's order, per dataset.
    assert plot.plot_model_selection("ds1", by_model) == ["B", "A"]
    assert plot.plot_model_selection("ds2", by_model) == ["A"]
    # No file: None, which leaves the plot to its usual top-n ranking.
    assert plot.plot_model_selection("ds3", by_model) is None


def test_plot_model_selection_warns_about_and_skips_an_unknown_name(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_model_order(tmp_path, "ds1", ["B", "Missing", "A"])
    kept = plot.plot_model_selection("ds1", {"A": ["rA"], "B": ["rB"]})
    assert kept == ["B", "A"]
    out = capsys.readouterr().out
    assert '"Missing"' in out and "no results" in out
    assert '"A"' not in out and '"B"' not in out


def test_model_order_file_leaves_rankings_and_tables_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_model_order(tmp_path, "ds1", ["A"])
    by_model = {
        "A": [{"conformal": {"P_latency": 40, "FPR": 0.0, "F1": 0.9}}],
        "B": [{"conformal": {"P_latency": 5, "FPR": 0.1, "F1": 0.5}}],
    }
    # The ranking helpers never read the file: every model stays, in metric order.
    assert report._order_models(by_model, "latency,fpr,f1") == ["B", "A"]
    assert report._order_rank(by_model, "latency,fpr,f1") == {"B": 0, "A": 1}
    assert report._top_models(by_model, "latency,fpr,f1", n=2) == ["B", "A"]


def test_read_model_order_file_rejects_anything_but_a_list_of_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_model_order(tmp_path, "ds1", {"order": ["TranAD"]})
    with pytest.raises(ValueError):
        plot._read_model_order_file(plot._model_order_path("ds1"))


def test_no_detection_and_its_dash():
    never = {"conformal": {"P_latency": 1734, "FPR": 0.0, "TP": 0, "FP": 0, "FN": 1734}}
    detected = {"conformal": {"P_latency": 12, "FPR": 0.0, "TP": 5, "FP": 0, "FN": 1}}
    assert metrics._no_detection(never, "conformal")
    assert not metrics._no_detection(detected, "conformal")
    assert metrics.format_metric(never, "conformal.P_latency", "conformal") == "---"
    assert metrics.format_metric(detected, "conformal.P_latency", "conformal") == "12"
    # Other columns are unaffected by a non-detection.
    assert metrics.format_metric(never, "conformal.FPR", "conformal") == "0"
