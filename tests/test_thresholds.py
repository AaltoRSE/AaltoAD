import json
import os

import numpy as np
import pandas as pd
import pytest

import AaltoAD.constants
import AaltoAD.pot as pot
from AaltoAD.thresholds import oracle as oracle_mod
from AaltoAD.thresholds import pot_fit
from AaltoAD.thresholds import scores as scores_mod
from AaltoAD.thresholds import shared as shared_mod


@pytest.fixture(autouse=True)
def initialize_constants():
    AaltoAD.constants.initialize(dataset='synthetic', model='TranAD')
    yield


class DummySPOT:
    """Same DummySPOT pattern used in tests/test_pot.py."""

    def __init__(self, q):
        self.q = q

    def fit(self, init_score_, score_):
        self.init = init_score_
        self.score = score_

    def initialize(self, level=None, min_extrema=False, verbose=False):
        self.level = level
        self.extreme_quantile = 0.6


# ---------------------------------------------------------------------------
# 1. fit_pot_threshold / pot_metrics reproduce pot.pot_eval
# ---------------------------------------------------------------------------

def test_fit_pot_threshold_and_pot_metrics_match_pot_eval(monkeypatch):
    init_score = np.array([0.1, 0.2, 0.05])
    score = np.array([0.2, 0.6, 0.1])
    label = np.array([0, 1, 0])

    monkeypatch.setattr(pot_fit, "SPOT", DummySPOT)

    expected, expected_preds = pot.pot_eval(init_score, score, label, expand_segments=False)
    expected_exp, expected_preds_exp = pot.pot_eval(init_score, score, label, expand_segments=True)

    threshold = pot_fit.fit_pot_threshold(init_score, score, q=1e-5, level=AaltoAD.constants.level)
    got = pot_fit.pot_metrics(score, label, threshold, expand_segments=False)
    got_exp = pot_fit.pot_metrics(score, label, threshold, expand_segments=True)

    assert threshold == pytest.approx(0.6)
    for k in expected:
        if k == 'p_latency' and expected[k] is None:
            assert got[k] is None
            continue
        assert got[k] == pytest.approx(expected[k], nan_ok=True)
    for k in expected_exp:
        if k == 'p_latency' and expected_exp[k] is None:
            assert got_exp[k] is None
            continue
        assert got_exp[k] == pytest.approx(expected_exp[k], nan_ok=True)


def test_fit_pot_threshold_nan_input_gives_nan():
    init_score = np.array([0.1, np.nan, 0.05])
    score = np.array([0.2, 0.6, 0.1])
    threshold = pot_fit.fit_pot_threshold(init_score, score, q=1e-5, level=AaltoAD.constants.level)
    assert np.isnan(threshold)


def test_pot_metrics_nan_threshold_gives_all_nan_dict():
    score = np.array([0.2, 0.6, 0.1])
    label = np.array([0, 1, 0])
    result = pot_fit.pot_metrics(score, label, float('nan'), expand_segments=False)
    for k in ('f1', 'precision', 'recall', 'fpr', 'TP', 'TN', 'FP', 'FN', 'ROC/AUC', 'threshold'):
        assert np.isnan(result[k])
    assert result['p_latency'] is None


# ---------------------------------------------------------------------------
# 2. oracle_metrics equals run_experiment.oracle_f1's block
# ---------------------------------------------------------------------------

def test_oracle_metrics_matches_run_experiment_oracle_f1():
    torch = pytest.importorskip('torch')
    from AaltoAD import run_experiment

    rng = np.random.default_rng(0)
    scores = rng.random(40)
    labels = np.zeros(40)
    labels[10:15] = 1
    labels[25:28] = 1

    for expand in (False, True):
        oracle_result, _ = run_experiment.oracle_f1(scores, labels, expand_segments=expand)
        assert oracle_result is not None
        got = oracle_mod.oracle_metrics(scores, labels, oracle_result['threshold'], expand_segments=expand)
        for k in oracle_result:
            if k == 'p_latency' and oracle_result[k] is None:
                assert got[k] is None
            else:
                assert got[k] == pytest.approx(oracle_result[k], nan_ok=True)


# ---------------------------------------------------------------------------
# 3. shared_oracle_threshold equals a brute-force sweep
# ---------------------------------------------------------------------------

def _make_series(seed, n=60):
    """Random scores/labels with 2-3 contiguous segments, none starting at row 0."""
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    labels = np.zeros(n)
    n_segments = rng.integers(2, 4)
    starts = rng.choice(range(5, n - 10), size=n_segments, replace=False)
    for s in sorted(starts):
        length = rng.integers(2, 5)
        labels[s:s + length] = 1
    return scores, labels


def _brute_force_pooled_threshold(scores_list, labels_list, expand_segments):
    """Sweep all unique scores, compute pooled F1 over datasets, return best (value, thr)."""
    candidates = np.unique(np.concatenate(scores_list))
    best_f1 = -1.0
    best_thr = None
    for thr in candidates:
        tp = fp = fn = 0
        for scores, labels in zip(scores_list, labels_list):
            m = oracle_mod.oracle_metrics(scores, labels, thr, expand_segments=expand_segments)
            tp += m['tp']
            fp += m['fp']
            fn += m['fn']
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        if f1 > best_f1 or (f1 == best_f1 and thr > best_thr):
            best_f1 = f1
            best_thr = thr
    return best_f1, best_thr


@pytest.mark.parametrize("expand_segments", [False, True])
def test_shared_oracle_threshold_matches_brute_force_sweep(expand_segments):
    scores_a, labels_a = _make_series(1)
    scores_b, labels_b = _make_series(2)
    scores_list = [scores_a, scores_b]
    labels_list = [labels_a, labels_b]

    expected_f1, expected_thr = _brute_force_pooled_threshold(scores_list, labels_list, expand_segments)

    got_thr = oracle_mod.shared_oracle_threshold(scores_list, labels_list, expand_segments=expand_segments)

    got_f1 = 0.0
    tp = fp = fn = 0
    for scores, labels in zip(scores_list, labels_list):
        m = oracle_mod.oracle_metrics(scores, labels, got_thr, expand_segments=expand_segments)
        tp += m['tp']
        fp += m['fp']
        fn += m['fn']
    denom = 2 * tp + fp + fn
    got_f1 = 0.0 if denom == 0 else 2 * tp / denom

    assert got_f1 == pytest.approx(expected_f1)
    # tie rule: the returned threshold is the highest among those tied for best F1
    assert got_thr == pytest.approx(expected_thr)


def test_shared_oracle_threshold_single_dataset_matches_oracle_f1():
    torch = pytest.importorskip('torch')
    from AaltoAD import run_experiment

    scores, labels = _make_series(3)

    for expand in (False, True):
        oracle_result, _ = run_experiment.oracle_f1(scores, labels, expand_segments=expand)
        assert oracle_result is not None

        got_thr = oracle_mod.shared_oracle_threshold([scores], [labels], expand_segments=expand)
        got = oracle_mod.oracle_metrics(scores, labels, got_thr, expand_segments=expand)

        assert got['f1'] == pytest.approx(oracle_result['f1'])


# ---------------------------------------------------------------------------
# 4. shared_threshold_blocks
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path, name, seed, hp_suffix="M_exp1.0"):
    ds_dir = tmp_path / name
    ds_dir.mkdir()

    scores, labels = _make_series(seed)
    calib_rng = np.random.default_rng(seed + 100)
    calib = calib_rng.random(30)

    results_path = ds_dir / f"{hp_suffix}_results.json"
    labels_path = ds_dir / f"{hp_suffix}_labels.csv"
    calib_path = ds_dir / f"{hp_suffix}_calib_scores.csv"

    with open(results_path, "w") as f:
        json.dump({"model": "M", "applied_hyperparameters": {}}, f)

    pd.DataFrame({
        "timestamp": np.arange(len(scores)),
        "ground_truth": labels,
        "prediction_error": scores,
    }).to_csv(labels_path, index=False)

    pd.DataFrame({"prediction_error": calib}).to_csv(calib_path, index=False)

    with open(results_path) as f:
        result = json.load(f)
    result["_source_path"] = str(results_path)
    return result, scores, labels, calib


def test_shared_threshold_blocks_identical_thresholds_and_summed_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(pot_fit, "SPOT", DummySPOT)

    result_a, scores_a, labels_a, calib_a = _write_dataset(tmp_path, "ds_a", seed=1)
    result_b, scores_b, labels_b, calib_b = _write_dataset(tmp_path, "ds_b", seed=2)

    results_by_dataset = {"ds_a": result_a, "ds_b": result_b}
    blocks = shared_mod.shared_threshold_blocks(results_by_dataset, q=1e-5, level=AaltoAD.constants.level)

    assert blocks is not None
    for ds in ("ds_a", "ds_b"):
        for block_name in ("pot", "pot_expanded", "oracle", "oracle_expanded"):
            assert block_name in blocks[ds]

    assert blocks["ds_a"]["pot"]["threshold"] == pytest.approx(blocks["ds_b"]["pot"]["threshold"])
    assert blocks["ds_a"]["oracle"]["threshold"] == pytest.approx(blocks["ds_b"]["oracle"]["threshold"])

    for method, tp_key, fp_key, fn_key in (
        ("pot", "TP", "FP", "FN"),
        ("oracle", "tp", "fp", "fn"),
    ):
        pooled_scores = np.concatenate([scores_a, scores_b])
        pooled_labels = np.concatenate([labels_a, labels_b])
        thr = blocks["ds_a"][method]["threshold"]
        if method == "pot":
            pooled_metrics = pot_fit.pot_metrics(pooled_scores, pooled_labels, thr, expand_segments=False)
        else:
            pooled_metrics = oracle_mod.oracle_metrics(pooled_scores, pooled_labels, thr, expand_segments=False)

        sum_tp = blocks["ds_a"][method][tp_key] + blocks["ds_b"][method][tp_key]
        sum_fp = blocks["ds_a"][method][fp_key] + blocks["ds_b"][method][fp_key]
        sum_fn = blocks["ds_a"][method][fn_key] + blocks["ds_b"][method][fn_key]
        assert sum_tp == pytest.approx(pooled_metrics[tp_key])
        assert sum_fp == pytest.approx(pooled_metrics[fp_key])
        assert sum_fn == pytest.approx(pooled_metrics[fn_key])


def test_shared_threshold_blocks_returns_none_when_csv_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pot_fit, "SPOT", DummySPOT)

    result_a, *_ = _write_dataset(tmp_path, "ds_a", seed=1)
    result_b, *_ = _write_dataset(tmp_path, "ds_b", seed=2)

    # remove one of ds_b's CSVs
    os.remove(str(tmp_path / "ds_b" / "M_exp1.0_calib_scores.csv"))

    blocks = shared_mod.shared_threshold_blocks(
        {"ds_a": result_a, "ds_b": result_b}, q=1e-5, level=AaltoAD.constants.level
    )
    assert blocks is None


# ---------------------------------------------------------------------------
# 5. Cache helpers
# ---------------------------------------------------------------------------

def test_cache_path_deterministic_independent_of_order(tmp_path):
    p1 = shared_mod.cache_path(str(tmp_path), ["ds_b", "ds_a"])
    p2 = shared_mod.cache_path(str(tmp_path), ["ds_a", "ds_b"])
    assert p1 == p2


def test_save_cache_then_load_cache_round_trips(tmp_path):
    path = shared_mod.cache_path(str(tmp_path), ["ds_a", "ds_b"])
    cache = {"M|{}": {"blocks": {"ds_a": {"pot": {"threshold": 0.5}}}, "sources": {"ds_a": [1.0, 2.0]}}}
    shared_mod.save_cache(path, cache)
    loaded = shared_mod.load_cache(path)
    assert loaded == cache


def test_load_cache_missing_path_returns_empty_dict(tmp_path):
    missing = str(tmp_path / "nonexistent" / "cache.json")
    assert shared_mod.load_cache(missing) == {}


# ---------------------------------------------------------------------------
# 6. report._apply_shared_thresholds
# ---------------------------------------------------------------------------

def test_apply_shared_thresholds_marks_results_and_leaves_incomplete_untouched(tmp_path, monkeypatch):
    import AaltoAD.report as report

    monkeypatch.setattr(pot_fit, "SPOT", DummySPOT)

    # POT_INIT resolves by prefix, so "synthetic_a"/"synthetic_b" match "synthetic".
    ds_a_name, ds_b_name = "synthetic_a", "synthetic_b"
    result_a, *_ = _write_dataset(tmp_path, ds_a_name, seed=1)
    result_b, *_ = _write_dataset(tmp_path, ds_b_name, seed=2)

    # A second configuration present only in ds_a (incomplete across datasets).
    ds_a_dir = tmp_path / ds_a_name
    only_a_path = ds_a_dir / "M_exp2.0_results.json"
    with open(only_a_path, "w") as f:
        json.dump({"model": "M", "applied_hyperparameters": {"q": 2.0}}, f)

    by_dataset = {
        ds_a_name: report._load_results(ds_a_name, results_folder=str(tmp_path)),
        ds_b_name: report._load_results(ds_b_name, results_folder=str(tmp_path)),
    }

    report._apply_shared_thresholds(by_dataset, results_folder=str(tmp_path))

    r_a = by_dataset[ds_a_name]["M"]
    r_b = by_dataset[ds_b_name]["M"]

    complete_a = [r for r in r_a if r.get("applied_hyperparameters") == {}][0]
    complete_b = [r for r in r_b if r.get("applied_hyperparameters") == {}][0]
    incomplete_a = [r for r in r_a if r.get("applied_hyperparameters") == {"q": 2.0}][0]

    assert complete_a.get("shared_threshold") is True
    assert complete_b.get("shared_threshold") is True
    assert complete_a["shared"]["pot"]["threshold"] == pytest.approx(complete_b["shared"]["pot"]["threshold"])
    assert complete_a["shared"]["oracle"]["threshold"] == pytest.approx(complete_b["shared"]["oracle"]["threshold"])
    # local blocks (from the results JSON) are not overwritten
    assert "pot" not in complete_a or complete_a["pot"] is not complete_a["shared"]["pot"]

    assert "shared_threshold" not in incomplete_a


def test_apply_shared_thresholds_second_call_hits_cache(tmp_path, monkeypatch):
    import AaltoAD.report as report
    from AaltoAD.thresholds import shared as shared_module

    monkeypatch.setattr(pot_fit, "SPOT", DummySPOT)

    ds_a_name, ds_b_name = "synthetic_a", "synthetic_b"
    _write_dataset(tmp_path, ds_a_name, seed=1)
    _write_dataset(tmp_path, ds_b_name, seed=2)

    by_dataset = {
        ds_a_name: report._load_results(ds_a_name, results_folder=str(tmp_path)),
        ds_b_name: report._load_results(ds_b_name, results_folder=str(tmp_path)),
    }
    report._apply_shared_thresholds(by_dataset, results_folder=str(tmp_path))

    call_count = {"n": 0}
    original = shared_module.shared_threshold_blocks

    def counting_fit(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_module, "shared_threshold_blocks", counting_fit)

    by_dataset2 = {
        ds_a_name: report._load_results(ds_a_name, results_folder=str(tmp_path)),
        ds_b_name: report._load_results(ds_b_name, results_folder=str(tmp_path)),
    }
    report._apply_shared_thresholds(by_dataset2, results_folder=str(tmp_path))

    assert call_count["n"] == 0


def test_group_configurations_groups_by_model_and_key_and_picks_best():
    r1 = {"applied_hyperparameters": {"a": 1}, "calibration_loss": 0.5}
    r2 = {"applied_hyperparameters": {"a": 1}, "calibration_loss": 0.1}
    r3 = {"applied_hyperparameters": {"a": 2}, "calibration_loss": 0.3}
    by_dataset = {"d1": {"M": [r1, r2]}, "d2": {"M": [r3]}}
    key = lambda r: json.dumps(r["applied_hyperparameters"], sort_keys=True)
    best = lambda rs: min(rs, key=lambda r: r["calibration_loss"])
    groups = shared_mod.group_configurations(by_dataset, key, best)
    assert set(groups) == {("M", key(r1)), ("M", key(r3))}
    assert groups[("M", key(r1))] == {"d1": r2}
    assert groups[("M", key(r3))] == {"d2": r3}


def test_source_mtimes_reads_csv_mtimes_and_none_when_missing(tmp_path):
    result_a, *_ = _write_dataset(tmp_path, "synthetic_a", seed=1)
    sources = shared_mod.source_mtimes({"synthetic_a": result_a})
    labels_path = result_a["_source_path"].replace("_results.json", "_labels.csv")
    assert sources["synthetic_a"][0] == os.path.getmtime(labels_path)
    assert shared_mod.source_mtimes({"x": {"_source_path": str(tmp_path / "nope_results.json")}}) is None
    assert shared_mod.source_mtimes({"x": {}}) is None


def test_shared_blocks_cached_fits_once_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pot_fit, "SPOT", DummySPOT)
    result_a, *_ = _write_dataset(tmp_path, "synthetic_a", seed=1)
    result_b, *_ = _write_dataset(tmp_path, "synthetic_b", seed=2)
    members = {"synthetic_a": result_a, "synthetic_b": result_b}
    calls = []
    real = shared_mod.shared_threshold_blocks
    monkeypatch.setattr(shared_mod, "shared_threshold_blocks",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = {}
    first = shared_mod.shared_blocks_cached(cache, "M", "{}", members, 1e-3, 0.98)
    second = shared_mod.shared_blocks_cached(cache, "M", "{}", members, 1e-3, 0.98)
    assert first == second and len(calls) == 1
    assert shared_mod.cache_key("M", "{}") in cache


def test_apply_blocks_stores_shared_blocks_and_keeps_local():
    r = {"pot": 1, "oracle": 2}
    blocks = {"d": {"pot": "p", "pot_expanded": "pe", "oracle": "o", "oracle_expanded": "oe"}}
    shared_mod.apply_blocks({"d": r}, blocks)
    assert r["pot"] == 1 and r["oracle"] == 2 and r["shared_threshold"] is True
    assert r["shared"] == {"pot": "p", "pot_expanded": "pe", "oracle": "o", "oracle_expanded": "oe"}


def test_with_blocks_swaps_method_blocks_or_returns_none():
    r = {"pot": 1, "oracle": 2, "calibration_loss": 0.5, "shared": {"pot": "p", "oracle": "o"}}
    view = shared_mod.with_blocks(r, "shared")
    assert view["pot"] == "p" and view["oracle"] == "o" and view["calibration_loss"] == 0.5
    assert r["pot"] == 1
    assert shared_mod.with_blocks({"pot": 1}, "shared") is None


def test_pooled_result_sums_counts_and_recomputes_metrics():
    from AaltoAD.thresholds.pooled import pooled_result
    r1 = {"model": "M", "dataset": "a", "experiment_id": "1.0", "calibration_loss": 1.0, "eval_time": 2.0,
          "shared_threshold": True, "applied_hyperparameters": {"q": 1},
          "pot": {"TP": 5, "FP": 5, "FN": 0, "TN": 10, "threshold": 0.3, "p_latency": 2.0},
          "oracle": {"tp": 5, "fp": 0, "fn": 5, "tn": 10, "threshold": 0.7, "p_latency": None}}
    r2 = {"model": "M", "dataset": "b", "experiment_id": "1.0", "calibration_loss": 3.0, "eval_time": 4.0,
          "shared_threshold": True, "applied_hyperparameters": {"q": 1},
          "pot": {"TP": 5, "FP": 0, "FN": 5, "TN": 10, "threshold": 0.3, "p_latency": 4.0},
          "oracle": {"tp": 10, "fp": 0, "fn": 0, "tn": 10, "threshold": 0.7, "p_latency": 1.0}}
    out = pooled_result([r1, r2])
    assert out["dataset"] == "combined" and out["datasets"] == ["a", "b"] and out["shared_threshold"]
    assert out["calibration_loss"] == 2.0 and out["eval_time"] == 6.0
    assert out["pot"]["TP"] == 10 and out["pot"]["FP"] == 5 and out["pot"]["FN"] == 5
    assert out["pot"]["f1"] == pytest.approx(2 * 10 / (2 * 10 + 5 + 5))
    assert out["pot"]["fpr"] == pytest.approx(5 / 25) and out["pot"]["threshold"] == 0.3
    assert out["pot"]["p_latency"] == 3.0 and out["oracle"]["p_latency"] == 1.0
    assert out["oracle"]["tp"] == 15 and out["oracle"]["threshold"] == 0.7
    assert "pot_expanded" not in out
    assert pooled_result([]) is None


def test_pooled_result_uses_blocks_key_when_present():
    from AaltoAD.thresholds.pooled import pooled_result
    r1 = {"model": "M", "dataset": "a", "shared_threshold": True,
          "pot": {"TP": 1, "FP": 0, "FN": 0, "TN": 0, "threshold": 0.1},
          "shared": {"pot": {"TP": 7, "FP": 0, "FN": 0, "TN": 0, "threshold": 0.9}}}
    r2 = {"model": "M", "dataset": "b", "shared_threshold": True,
          "pot": {"TP": 1, "FP": 0, "FN": 0, "TN": 0, "threshold": 0.1},
          "shared": {"pot": {"TP": 3, "FP": 0, "FN": 0, "TN": 0, "threshold": 0.9}}}
    assert pooled_result([r1, r2])["pot"]["TP"] == 2
    out = pooled_result([r1, r2], blocks_key="shared")
    assert out["pot"]["TP"] == 10 and out["pot"]["threshold"] == 0.9
