"""Metric lookup, formatting, and comparison for sweep results."""

import json
import math

from AaltoAD.report import cli

LOWER_IS_BETTER = {"p_latency", "fpr", "threshold", "calibration_loss"}


TOP_LEVEL_METRICS = {
    "calibration_loss",
    "eval_time",
}  # metrics that are not in the method block


SUMMARY_SORT_METRIC = "pot.f1"


def _metric_path(name, method):
    """Full lookup path for a bare metric name: 'f1' -> '<method>.f1'."""
    if "." in name or name in TOP_LEVEL_METRICS:
        return name
    return f"{method}.{name}"


def _no_detection(result, method):
    """True when `result`'s `method` block has no true positives.

    `segment_latency` charges an undetected segment its full length, so such a
    latency records how long the segment was, not how long detection took. It
    must not be ranked or printed as a detection time — and its value varies
    with the model's warm-up, which makes it look faster the more rows a model
    drops before scoring.
    """
    counts = _confusion_counts(result, method) if result else None
    return counts is not None and counts[0] == 0


def _fmt_metric(result, path, method):
    """Format one metric cell, as an em dash for a latency that is really a non-detection."""
    if (
        isinstance(path, str)
        and path.endswith(".p_latency")
        and _no_detection(result, method)
    ):
        return "---"
    return _fmt(_col_value(result, path))


def _rank_key(result, metric, method):
    """Sort key ranking one result by `metric`, best first.

    `metric` is a tuple of bare names (see `cli.metric_list`) read from the `method`
    block, each in its natural direction. A missing value sorts last within its
    own term, and so does the latency of a model that never fired (see
    `_no_detection`), so a model that never fires cannot outrank one that does.
    """
    key = []
    for name in cli.metric_list(metric):
        value = _metric_value(result, _metric_path(name, method)) if result else None
        if name == "p_latency" and _no_detection(result, method):
            value = None
        if value is None:
            key.append(float("inf"))
        else:
            key.append(value if name in LOWER_IS_BETTER else -value)
    return tuple(key)


def _get(d, dotted_key):
    """Look up a possibly-dotted key like 'pot.f1' in a nested dict, returning None if absent."""
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _metric_value(result, metric):
    """Return the metric as a float, or None if absent/non-numeric/NaN."""
    try:
        v = float(_get(result, metric))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _hp_key(result):
    """Canonical key identifying a hyperparameter configuration."""
    return json.dumps(result.get("applied_hyperparameters", {}), sort_keys=True)


def _confusion_counts(result, method):
    """Return (TP, FP, FN) for a method block, or None if any is absent.

    POT blocks store upper-case keys, oracle blocks lower-case ones.
    """
    block = result.get(method)
    if not isinstance(block, dict):
        return None
    out = []
    for name in ("TP", "FP", "FN"):
        v = block.get(name, block.get(name.lower()))
        if v is None:
            return None
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return tuple(out)


def _pooled_f1(results, method):
    """F1 recomputed from confusion counts summed over `results`.

    Returns None if any result lacks counts for `method`.
    """
    tp = fp = fn = 0.0
    for r in results:
        counts = _confusion_counts(r, method)
        if counts is None:
            return None
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _select_shared_best(by_dataset, metric, method=None):
    """Select one hyperparameter set per model across all datasets.

    For each model, group results by hyperparameter configuration (keeping the
    best result per dataset within a configuration) and pick the configuration
    with the best score over the listed datasets. Each term of `metric` is
    aggregated across the datasets: an F1 is recomputed from pooled confusion
    counts, since averaging F1 values is not meaningful, and everything else is
    averaged. Only configurations with a run in every dataset are eligible; if a
    model has none, it falls back to the best score over the available runs and
    warns.

    Returns {dataset: {model: [result]}} with at most one result per model; an
    empty list (with a warning) marks a dataset where the selected
    configuration has no run.
    """
    method = method or cli.THRESHOLD_METHOD
    metric = cli.metric_list(metric)
    datasets = list(by_dataset)
    per_model = {}  # model -> hp_key -> {dataset: result}
    for ds, by_model in by_dataset.items():
        for model, results in by_model.items():
            for r in results:
                slot = per_model.setdefault(model, {}).setdefault(_hp_key(r), {})
                if ds not in slot or _rank_key(r, metric, method) < _rank_key(
                    slot[ds], metric, method
                ):
                    slot[ds] = r

    selected = {ds: {} for ds in datasets}
    for model, configs in per_model.items():
        complete = {k: v for k, v in configs.items() if len(v) == len(datasets)}
        pool = complete or configs
        if not complete:
            print(
                f"Warning: {model}: no hyperparameter set has runs in all "
                f"datasets; selecting by score over available runs."
            )

        def _score(key):
            """Aggregate each metric term over the datasets, best first."""
            results = list(pool[key].values())
            score = []
            for name in metric:
                if name == "f1":
                    # An average of F1 values is not meaningful; recompute it
                    # from the confusion counts pooled over the datasets.
                    pooled = _pooled_f1(results, method)
                    score.append(float("inf") if pooled is None else -pooled)
                    continue
                values = [_metric_value(r, _metric_path(name, method)) for r in results]
                values = [v for v in values if v is not None]
                if not values:
                    score.append(float("inf"))
                else:
                    mean = sum(values) / len(values)
                    score.append(mean if name in LOWER_IS_BETTER else -mean)
            return tuple(score)

        best_key = min(pool, key=_score)
        for ds in datasets:
            entry = configs[best_key].get(ds)
            if entry is None:
                print(
                    f"Warning: {model}: selected hyperparameters have no run "
                    f"for dataset {ds}."
                )
                selected[ds][model] = []
            else:
                selected[ds][model] = [entry]
    return selected


def _best_result(results, metric, method=None):
    """Return the best of `results` by `metric`, read from the `method` block.

    Ranking is `_rank_key`, so a multi-part metric such as ('p_latency', 'fpr',
    'f1') is resolved term by term. A result missing every term ranks last but
    is still returned when it is all there is, so its other columns render
    instead of blanking the row to N/A.
    """
    if not results:
        return None
    method = method or cli.THRESHOLD_METHOD
    return min(results, key=lambda r: _rank_key(r, metric, method))


def _fmt(value):
    """Format a numeric value for display (scientific notation below 1e-3)."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if math.isnan(v):
            return "NaN"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        if abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:.4f}"
    except (TypeError, ValueError):
        return str(value)


def _is_unlabeled(result):
    """Return True when a result has no positive labels (pot.TP + pot.FN == 0).

    Returns False if the pot section or the required keys are absent.
    """
    pot = result.get("pot")
    if not isinstance(pot, dict):
        return False
    tp = pot.get("TP")
    fn = pot.get("FN")
    if tp is None or fn is None:
        return False
    try:
        return float(tp) + float(fn) == 0
    except (TypeError, ValueError):
        return False


def _detected(result):
    """Return detected count (pot.TP + pot.FP), or None if keys absent."""
    pot = result.get("pot")
    if not isinstance(pot, dict):
        return None
    tp = pot.get("TP")
    fp = pot.get("FP")
    if tp is None or fp is None:
        return None
    try:
        return float(tp) + float(fp)
    except (TypeError, ValueError):
        return None


def _detection_rate(result):
    """Return detected / total (TP+FP+TN+FN), or None if keys absent or denominator zero."""
    pot = result.get("pot")
    if not isinstance(pot, dict):
        return None
    tp = pot.get("TP")
    fp = pot.get("FP")
    tn = pot.get("TN")
    fn = pot.get("FN")
    if any(v is None for v in (tp, fp, tn, fn)):
        return None
    try:
        total = float(tp) + float(fp) + float(tn) + float(fn)
        if total == 0:
            return None
        return (float(tp) + float(fp)) / total
    except (TypeError, ValueError):
        return None


def _col_value(result, path_or_fn):
    """Return a column value from a result dict.

    `path_or_fn` may be a dotted path string or a callable taking the result dict.
    """
    if callable(path_or_fn):
        return path_or_fn(result)
    return _get(result, path_or_fn)


def _table_sort_metric(metric):
    """Fallback single metric for the unlabeled tables, which cannot rank by detections."""
    metric = metric if isinstance(metric, str) else ",".join(metric)
    return metric if "." in metric else SUMMARY_SORT_METRIC
