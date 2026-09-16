"""Fit POT/oracle thresholds shared across datasets for one configuration, with a JSON cache."""

import json
import os

import numpy as np

from AaltoAD.thresholds.scores import load_run_scores
from AaltoAD.thresholds.pot_fit import fit_pot_threshold, pot_metrics
from AaltoAD.thresholds.conformal import conformal_metrics, conformal_threshold
from AaltoAD.thresholds.oracle import oracle_metrics, shared_oracle_threshold

# Bumped whenever the contents of a blocks dict change, so cached entries
# written by an older version are refitted instead of silently reused.
BLOCKS_VERSION = 3


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


def shared_threshold_blocks(results_by_dataset, q, level, conformal_q=None):
    """Fit and evaluate shared thresholds for one model/hyperparameter configuration.

    `results_by_dataset` is ``{dataset: result_dict}``, one result per
    dataset for the same configuration. Loads each dataset's calibration/test
    scores and labels, pools calibration scores to fit one POT threshold
    (``fit_pot_threshold``, at the run's swept risk `q`) and one conformal
    threshold (``conformal_threshold`` at `conformal_q`, defaulting to `q` when
    it is None), and concatenates test
    scores/labels to search one raw and one segment-expanded oracle threshold
    (``shared_oracle_threshold``). Per dataset, evaluates every block at the
    shared thresholds.

    Returns ``{dataset: {'conformal': ..., 'conformal_expanded': ..., 'pot':
    ..., 'pot_expanded': ..., 'oracle': ..., 'oracle_expanded': ...}}`` with
    plain, JSON-serialisable Python numeric types, or ``None`` if any dataset's
    score CSVs could not be loaded.
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
    conformal_thr = conformal_threshold(pooled_calib, q if conformal_q is None else conformal_q)

    test_list = [test for _, test, _ in per_dataset_scores.values()]
    label_list = [labels for _, _, labels in per_dataset_scores.values()]
    oracle_threshold_raw = shared_oracle_threshold(test_list, label_list, expand_segments=False)
    oracle_threshold_expanded = shared_oracle_threshold(test_list, label_list, expand_segments=True)

    blocks = {}
    for ds, (_, test, labels) in per_dataset_scores.items():
        blocks[ds] = {
            'conformal': conformal_metrics(test, labels, conformal_thr, False),
            'conformal_expanded': conformal_metrics(test, labels, conformal_thr, True),
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


def shared_blocks_cached(cache, model, hp_key, results_by_dataset, q, level, conformal_q=None):
    """Return shared-threshold blocks for one configuration, fitting only on a cache miss.

    A cache entry is reused when its recorded CSV modification times, block
    version and `conformal_q` all match; otherwise ``shared_threshold_blocks``
    is called and the result stored in `cache` (mutated in place). Returns the
    blocks, or ``None`` if the source CSVs are missing or the fit could not be
    done.
    """
    sources = source_mtimes(results_by_dataset)
    if sources is None:
        return None
    key = cache_key(model, hp_key)
    entry = cache.get(key)
    if (entry and entry.get("sources") == sources
            and entry.get("version") == BLOCKS_VERSION
            and entry.get("conformal_q") == conformal_q):
        return entry["blocks"]
    blocks = shared_threshold_blocks(results_by_dataset, q, level, conformal_q)
    if blocks is not None:
        cache[key] = {"blocks": blocks, "sources": sources, "version": BLOCKS_VERSION,
                      "conformal_q": conformal_q}
    return blocks


METHOD_BLOCKS = ("conformal", "conformal_expanded", "pot", "pot_expanded", "oracle", "oracle_expanded")

# A conformal threshold is defined by the calibration sample it is fitted on,
# and the report fits it on the pooled calibration of every dataset it covers.
# There is therefore no meaningful per-dataset conformal block to preserve: the
# shared fit *is* the conformal result, so it is also written to the top level,
# where --threshold-method conformal reads it.
SHARED_ONLY_BLOCKS = ("conformal", "conformal_expanded")


def apply_blocks(results_by_dataset, blocks):
    """Store the shared blocks under ``result["shared"]`` and mark the result.

    The local (per-dataset) pot/oracle blocks stay in place;
    ``with_blocks(result, "shared")`` yields a view where the shared ones
    replace them. `SHARED_ONLY_BLOCKS` are additionally written to the top
    level, since they have no local counterpart.
    """
    for ds, r in results_by_dataset.items():
        r["shared"] = {name: blocks[ds][name] for name in METHOD_BLOCKS if name in blocks[ds]}
        r["shared_threshold"] = True
        for name in SHARED_ONLY_BLOCKS:
            if name in blocks[ds]:
                r[name] = blocks[ds][name]


def with_blocks(result, key):
    """Shallow copy of `result` with its four method blocks taken from ``result[key]``.

    Returns ``None`` if `result` has no ``key`` entry.
    """
    alt = result.get(key)
    if not isinstance(alt, dict):
        return None
    view = dict(result)
    for name in METHOD_BLOCKS:
        if name in alt:
            view[name] = alt[name]
    return view
