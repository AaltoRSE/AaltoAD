"""Fit POT/oracle thresholds shared across datasets for one configuration, with a JSON cache."""

import json
import os

import numpy as np

from AaltoAD.thresholds.scores import load_run_scores
from AaltoAD.thresholds.pot_fit import fit_pot_threshold, pot_metrics
from AaltoAD.thresholds.oracle import oracle_metrics, shared_oracle_threshold


def _to_jsonable(value):
    """Recursively convert numpy scalar/array values inside `value` to native Python types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def shared_threshold_blocks(results_by_dataset, q, level):
    """Fit and evaluate shared thresholds for one model/hyperparameter configuration.

    `results_by_dataset` is ``{dataset: result_dict}``, one result per
    dataset for the same configuration. Loads each dataset's calibration/test
    scores and labels, pools calibration scores to fit one POT threshold
    (``fit_pot_threshold``), and concatenates test scores/labels to search
    one raw and one segment-expanded oracle threshold
    (``shared_oracle_threshold``). Per dataset, evaluates all four blocks
    (``pot_metrics`` / ``oracle_metrics``) at the shared thresholds.

    Returns ``{dataset: {'pot': ..., 'pot_expanded': ..., 'oracle': ...,
    'oracle_expanded': ...}}`` with plain, JSON-serialisable Python numeric
    types, or ``None`` if any dataset's score CSVs could not be loaded.
    """
    per_dataset_scores = {}
    for ds, result in results_by_dataset.items():
        loaded = load_run_scores(result)
        if loaded is None:
            return None
        per_dataset_scores[ds] = loaded

    pooled_calib = np.concatenate([calib for calib, _, _ in per_dataset_scores.values()])
    pooled_test = np.concatenate([test for _, test, _ in per_dataset_scores.values()])
    pot_threshold = fit_pot_threshold(pooled_calib, pooled_test, q, level)

    test_list = [test for _, test, _ in per_dataset_scores.values()]
    label_list = [labels for _, _, labels in per_dataset_scores.values()]
    oracle_threshold_raw = shared_oracle_threshold(test_list, label_list, expand_segments=False)
    oracle_threshold_expanded = shared_oracle_threshold(test_list, label_list, expand_segments=True)

    blocks = {}
    for ds, (_, test, labels) in per_dataset_scores.items():
        blocks[ds] = {
            'pot': pot_metrics(test, labels, pot_threshold, False),
            'pot_expanded': pot_metrics(test, labels, pot_threshold, True),
            'oracle': oracle_metrics(test, labels, oracle_threshold_raw, False),
            'oracle_expanded': oracle_metrics(test, labels, oracle_threshold_expanded, True),
        }
    return _to_jsonable(blocks)


def cache_path(results_folder, datasets):
    """Path to the shared-threshold cache JSON for this exact set of datasets."""
    name = "+".join(sorted(datasets)) + ".json"
    return os.path.join(results_folder, "_shared_thresholds", name)


def load_cache(path):
    """Load the shared-threshold cache dict from `path`; return {} if missing/unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_cache(path, cache):
    """Write the shared-threshold cache dict to `path`, creating parent directories."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


def cache_key(model, hp_key):
    """Cache key identifying one model/hyperparameter-configuration entry."""
    return f"{model}|{hp_key}"


def group_configurations(by_dataset, hp_key, pick_best):
    """Group results by (model, hyperparameter key) across datasets.

    `by_dataset` is ``{dataset: {model: [result, ...]}}`` as loaded by the
    report. `hp_key(result)` returns the configuration key; when a dataset has
    several results with the same key, `pick_best(results)` chooses one.
    Returns ``{(model, key): {dataset: result}}``.
    """
    groups = {}
    for ds, by_model in by_dataset.items():
        for model, results in by_model.items():
            by_hp = {}
            for r in results:
                by_hp.setdefault(hp_key(r), []).append(r)
            for key, candidates in by_hp.items():
                chosen = candidates[0] if len(candidates) == 1 else pick_best(candidates)
                groups.setdefault((model, key), {})[ds] = chosen
    return groups


def source_mtimes(results_by_dataset):
    """Modification times of the labels/calibration CSVs behind each result.

    Returns ``{dataset: [labels_mtime, calib_mtime]}``, or ``None`` if any
    result lacks a source path or either CSV is missing. Used to decide
    whether a cached shared-threshold entry is still valid.
    """
    sources = {}
    for ds, r in results_by_dataset.items():
        src = r.get("_source_path")
        if not src:
            return None
        paths = [src.replace("_results.json", "_labels.csv"),
                 src.replace("_results.json", "_calib_scores.csv")]
        if not all(os.path.exists(p) for p in paths):
            return None
        sources[ds] = [os.path.getmtime(p) for p in paths]
    return sources


def shared_blocks_cached(cache, model, hp_key, results_by_dataset, q, level):
    """Return shared-threshold blocks for one configuration, fitting only on a cache miss.

    A cache entry is reused when its recorded CSV modification times match
    the current files; otherwise ``shared_threshold_blocks`` is called and
    the result stored in `cache` (mutated in place). Returns the blocks, or
    ``None`` if the source CSVs are missing or the fit could not be done.
    """
    sources = source_mtimes(results_by_dataset)
    if sources is None:
        return None
    key = cache_key(model, hp_key)
    entry = cache.get(key)
    if entry and entry.get("sources") == sources:
        return entry["blocks"]
    blocks = shared_threshold_blocks(results_by_dataset, q, level)
    if blocks is not None:
        cache[key] = {"blocks": blocks, "sources": sources}
    return blocks


def apply_blocks(results_by_dataset, blocks):
    """Overwrite each result's pot/oracle blocks with the shared ones and mark it shared."""
    for ds, r in results_by_dataset.items():
        for name in ("pot", "pot_expanded", "oracle", "oracle_expanded"):
            r[name] = blocks[ds][name]
        r["shared_threshold"] = True
