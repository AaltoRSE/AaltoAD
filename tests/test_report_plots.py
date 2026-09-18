import pandas as pd
import pytest

import AaltoAD.report as report


def test_method_name_normalises_dotted_metrics():
    assert report.method_name("conformal") == "conformal"
    assert report.method_name("conformal.f1") == "conformal"
    assert report.method_name("oracle_expanded.f1") == "oracle"
    assert report.method_name("pot.threshold") == "pot"


def test_plot_threshold_picks_method_threshold_and_rejects_invalid():
    result = {"pot": {"threshold": 0.5}, "conformal": {"threshold": 0.2}}
    assert report._plot_threshold(result, "pot") == 0.5
    assert report._plot_threshold(result, "conformal") == 0.2
    assert report._plot_threshold({"pot": {"threshold": 0.0}}, "pot") is None
    assert report._plot_threshold({}, "conformal") is None


def test_clip_for_plot_caps_values_at_the_shared_scale():
    values = pd.Series([0.5, 1.0, 2.5, 100.0], index=[-1, 0, 1, 2])
    clipped = report._clip_for_plot(values)
    assert clipped.max() == report.PLOT_VALUE_CLIP
    assert clipped.tolist() == [0.5, 1.0, 2.0, 2.0]


def test_plot_rank_key_follows_the_given_order():
    slow_but_accurate = {"pot": {"p_latency": 50, "fpr": 0.0, "f1": 0.9}}
    fast_but_poor = {"pot": {"p_latency": 5, "fpr": 0.1, "f1": 0.1}}
    by_latency = ("p_latency", "fpr", "f1")
    by_f1 = ("f1",)
    assert report._rank_key(fast_but_poor, by_latency, "pot") < \
        report._rank_key(slow_but_accurate, by_latency, "pot")
    assert report._rank_key(slow_but_accurate, by_f1, "pot") < \
        report._rank_key(fast_but_poor, by_f1, "pot")


def test_plot_rank_key_prefers_latency_then_fpr_then_f1():
    fast = {"pot": {"p_latency": 5, "fpr": 0.1, "f1": 0.1}}
    slow = {"pot": {"p_latency": 50, "fpr": 0.0, "f1": 0.9}}
    assert report._rank_key(fast, ("p_latency", "fpr", "f1"), "pot") < report._rank_key(slow, ("p_latency", "fpr", "f1"), "pot")


    clean = {"pot": {"p_latency": 5, "fpr": 0.01, "f1": 0.1}}
    noisy = {"pot": {"p_latency": 5, "fpr": 0.2, "f1": 0.9}}
    assert report._rank_key(clean, ("p_latency", "fpr", "f1"), "pot") < report._rank_key(noisy, ("p_latency", "fpr", "f1"), "pot")

    good = {"pot": {"p_latency": 5, "fpr": 0.01, "f1": 0.9}}
    poor = {"pot": {"p_latency": 5, "fpr": 0.01, "f1": 0.2}}
    assert report._rank_key(good, ("p_latency", "fpr", "f1"), "pot") < report._rank_key(poor, ("p_latency", "fpr", "f1"), "pot")


def test_plot_rank_key_sorts_missing_values_last():
    present = {"pot": {"p_latency": 1000, "fpr": 0.9, "f1": 0.0}}
    missing = {"pot": {}}
    assert report._rank_key(present, ("p_latency", "fpr", "f1"), "pot") < report._rank_key(missing, ("p_latency", "fpr", "f1"), "pot")
    assert report._rank_key(None, ("p_latency", "fpr", "f1"), "pot") == report._rank_key(missing, ("p_latency", "fpr", "f1"), "pot")


def test_top_models_ranks_by_latency_then_fpr_then_f1():
    by_model = {
        # A detects slowest, B fastest; C never fires and has no metrics at all.
        "A": [{"conformal": {"p_latency": 40, "fpr": 0.0, "f1": 0.9}}],
        "B": [{"conformal": {"p_latency": 5, "fpr": 0.1, "f1": 0.5}}],
        "C": [{"calibration_loss": 2.0}],
    }
    assert report._top_models(by_model, "latency,fpr,f1", n=2) == ["B", "A"]
    assert report._top_models(by_model, "latency,fpr,f1", n=3)[-1] == "C"


def test_top_models_reads_the_threshold_method_it_is_given():
    by_model = {
        "A": [{"pot": {"p_latency": 90, "fpr": 0.0, "f1": 0.1},
               "conformal": {"p_latency": 1, "fpr": 0.0, "f1": 0.9}}],
        "B": [{"pot": {"p_latency": 2, "fpr": 0.0, "f1": 0.1},
               "conformal": {"p_latency": 80, "fpr": 0.0, "f1": 0.9}}],
    }
    assert report._top_models(by_model, "latency,fpr,f1", n=2, threshold_method="pot") == ["B", "A"]
    assert report._top_models(by_model, "latency,fpr,f1", n=2, threshold_method="conformal") == ["A", "B"]


def test_top_models_honours_a_custom_model_order():
    by_model = {
        "A": [{"conformal": {"p_latency": 40, "fpr": 0.0, "f1": 0.9}}],
        "B": [{"conformal": {"p_latency": 5, "fpr": 0.1, "f1": 0.5}}],
    }
    assert report._top_models(by_model, "latency,fpr,f1", n=2, model_order=("f1",)) == ["A", "B"]
    assert report._top_models(by_model, "latency,fpr,f1", n=2, model_order=("fpr",)) == ["A", "B"]
    assert report._top_models(by_model, "latency,fpr,f1", n=2, model_order=("p_latency",)) == ["B", "A"]


def test_order_rank_gives_positions_in_display_order():
    by_model = {
        "slow": [{"conformal": {"p_latency": 40, "fpr": 0.0, "f1": 0.9}}],
        "fast": [{"conformal": {"p_latency": 5, "fpr": 0.1, "f1": 0.5}}],
    }
    assert report._order_rank(by_model, "latency,fpr,f1") == {"fast": 0, "slow": 1}


def test_sort_rows_follows_the_order_when_given():
    rows = [(0.9, {"model": "slow"}), (0.5, {"model": "fast"})]
    report._sort_rows(rows, {"fast": 0, "slow": 1}, lower_is_better=False)
    assert [r["model"] for _, r in rows] == ["fast", "slow"]
    # Without an order, rows fall back to their own metric value.
    report._sort_rows(rows, None, lower_is_better=False)
    assert [r["model"] for _, r in rows] == ["slow", "fast"]


def test_metric_list_accepts_the_spellings_the_cli_allows():
    assert report.metric_list("latency,fpr,f1") == ("p_latency", "fpr", "f1")
    assert report.metric_list("conformal.f1") == ("f1",)
    assert report.metric_list(("p_latency", "f1")) == ("p_latency", "f1")
    assert report.metric_list("calibration_loss") == ("calibration_loss",)


def test_metric_path_keeps_top_level_metrics_out_of_the_method_block():
    assert report._metric_path("f1", "conformal") == "conformal.f1"
    assert report._metric_path("calibration_loss", "conformal") == "calibration_loss"


def test_latex_columns_default_to_the_paper_set():
    assert report._latex_columns("conformal") == [
        ("F1", "conformal.f1"),
        ("FPR", "conformal.fpr"),
        ("Latency", "conformal.p_latency"),
    ]


def test_latex_columns_take_the_names_they_are_given():
    columns = report._latex_columns("pot", "f1,adjusted_f1,eval_time")
    assert columns == [
        ("F1", "pot.f1"),
        ("Adjusted F1", "pot_expanded.f1"),
        ("Eval time (s)", "eval_time"),
    ]
    # An unlisted name is read from the method's own block.
    assert report._latex_columns("pot", "recall")[0] == ("Recall", "pot.recall")


def test_latex_tabular_deals_rows_into_blocks_column_major():
    rows = [{"model": f"M{i}"} for i in range(5)]
    table = report._latex_tabular(["model"], rows, blocks=2)
    assert r"\begin{tabular}{l@{\qquad}l}" in table
    assert r"model & model \\" in table
    # Five rows over two blocks: three lines, the last with an empty right cell.
    assert r"M0 & M3 \\" in table
    assert r"M2 &  \\" in table


def test_blocked_rows_keeps_order_within_a_block():
    rows = list(range(6))
    assert report._blocked_rows(rows, 2) == [(0, 3), (1, 4), (2, 5)]
    assert report._blocked_rows(rows, 1) == [(0,), (1,), (2,), (3,), (4,), (5,)]


def test_rank_key_ignores_the_latency_of_a_model_that_never_fired():
    # Both "detect" at step 100, but `never` has no true positives, so its
    # latency is just the segment length and must not win the comparison.
    detected = {"conformal": {"p_latency": 1790, "fpr": 0.0, "f1": 0.01, "TP": 5, "FP": 0, "FN": 10}}
    never = {"conformal": {"p_latency": 1734, "fpr": 0.0, "f1": 0.0, "TP": 0, "FP": 0, "FN": 1734}}
    order = ("p_latency", "fpr", "f1")
    assert report._rank_key(detected, order, "conformal") < report._rank_key(never, order, "conformal")


def test_no_detection_and_its_dash():
    never = {"conformal": {"p_latency": 1734, "fpr": 0.0, "TP": 0, "FP": 0, "FN": 1734}}
    detected = {"conformal": {"p_latency": 12, "fpr": 0.0, "TP": 5, "FP": 0, "FN": 1}}
    assert report._no_detection(never, "conformal")
    assert not report._no_detection(detected, "conformal")
    assert report._fmt_metric(never, "conformal.p_latency", "conformal") == "---"
    assert report._fmt_metric(detected, "conformal.p_latency", "conformal") == "12"
    # Other columns are unaffected by a non-detection.
    assert report._fmt_metric(never, "conformal.fpr", "conformal") == "0"
