"""Loading of sweep result JSON files for the report pipeline.

`load_results` reads every dataset's ``*_results.json`` files into the
by-dataset/by-model mapping the report pipeline works on, and fits/attaches
the threshold blocks (cached by CSV modification times).
"""

import json
import os
from glob import glob

from tqdm import tqdm

from AaltoAD import constants
from AaltoAD.report import cli, metrics
from AaltoAD.thresholds import shared


def load_results(
    datasets,
    results_folder="results",
    conformal_q=constants.CONFORMAL_Q,
    pool_baselines=cli.POOL_BASELINES,
):
    """Load the result JSONs of every dataset and attach the threshold blocks.

    Returns ``{dataset: {model: [results]}}``; datasets without result files
    are left out (with a message). The thresholds are fitted here because the
    conformal threshold is defined by the calibration sample it is fitted on;
    with `pool_baselines` the calibration of every dataset is pooled, by
    default each dataset is fitted on its own baseline.
    """
    by_dataset = {}
    for dataset in datasets:
        by_model = _load_dataset(dataset, results_folder)
        if not by_model:
            print(f'No results found for dataset "{dataset}" in {results_folder}/')
            continue
        by_dataset[dataset] = by_model
    if by_dataset:
        _apply_thresholds(by_dataset, results_folder, conformal_q, pool_baselines)
    return by_dataset


def _load_dataset(dataset, results_folder):
    """Load all result JSONs for a dataset, grouped by model."""
    pattern = os.path.join(results_folder, dataset, "*_results.json")
    files = glob(pattern)
    by_model = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        # Remember where this result came from so per-model plots can locate the
        # matching *_labels.csv (same path with _results.json -> _labels.csv).
        data["_source_path"] = path
        model = data.get("model", "unknown")
        by_model.setdefault(model, []).append(data)
    return by_model


def _apply_thresholds(
    by_dataset,
    results_folder,
    conformal_q=constants.CONFORMAL_Q,
    pool_baselines=cli.POOL_BASELINES,
):
    """Attach conformal/pot/oracle blocks to every result, in place.

    Thresholds are shared across datasets only with `pool_baselines` (the
    ``--pool-baselines`` CLI flag): one threshold per configuration is then
    fit on the calibration data of every dataset pooled together — which
    assumes the baselines are alike, and lets a contaminated one set the
    threshold for all the rest. By default each dataset is fit on its own
    baseline instead, by
    running the same machinery once per dataset; the blocks still land where
    the pooled ones would, so the combined report and the shared plots keep
    working on separately fitted thresholds.
    """
    if not pool_baselines and len(by_dataset) > 1:
        for dataset, by_model in by_dataset.items():
            _fit_threshold_blocks({dataset: by_model}, results_folder, conformal_q)
        return
    _fit_threshold_blocks(by_dataset, results_folder, conformal_q)


def _fit_threshold_blocks(
    by_dataset, results_folder, conformal_q=constants.CONFORMAL_Q
):
    """Fit and attach threshold blocks over the given datasets, pooling their calibration.

    Only configurations (model + hyperparameters) with a run in every dataset
    are processed; per dataset the lowest-``calibration_loss`` run represents
    the configuration. Each processed result gets its shared blocks under
    ``result["shared"]`` and ``shared_threshold = True``; the local pot/oracle
    blocks stay untouched and remain what per-dataset tables show, while the
    conformal blocks, which have no local counterpart, are written to the top
    level as well (see ``shared.apply_blocks``). POT uses each configuration's
    swept ``q``; the conformal threshold uses `conformal_q`. Fits are cached
    under ``results_folder/_shared_thresholds/`` keyed by the CSV modification
    times. With a single dataset the "pool" is that dataset's own calibration,
    which is how `_apply_thresholds` fits separate baselines.
    """

    datasets = list(by_dataset)
    groups = shared.group_configurations(
        by_dataset,
        metrics._hp_key,
        lambda rs: metrics._best_result(rs, "calibration_loss"),
    )
    complete = {k: v for k, v in groups.items() if set(v) == set(datasets)}
    print(
        f"Fitting thresholds for {len(complete)} configurations on the calibration of {datasets}"
    )

    cache_file = shared.cache_path(results_folder, datasets)
    cache = shared.load_cache(cache_file)
    shared_count, skipped_count = {}, {}
    for (model, _), members in groups.items():
        if set(members) != set(datasets):
            skipped_count[model] = skipped_count.get(model, 0) + 1

    for (model, hp_key), results_by_dataset in tqdm(
        complete.items(), desc="shared thresholds", unit="config"
    ):
        q = (
            next(iter(results_by_dataset.values()))
            .get("applied_hyperparameters", {})
            .get("q", 1e-5)
        )
        constants.initialize(datasets[0], model)
        blocks = shared.shared_blocks_cached(
            cache, model, hp_key, results_by_dataset, q, constants.level, conformal_q
        )
        if blocks is None:
            skipped_count[model] = skipped_count.get(model, 0) + 1
            continue
        shared.apply_blocks(results_by_dataset, blocks)
        shared_count[model] = shared_count.get(model, 0) + 1

    shared.save_cache(cache_file, cache)
    for model in sorted(set(shared_count) | set(skipped_count)):
        print(
            f"  {model}: {shared_count.get(model, 0)} configuration(s) shared, "
            f"{skipped_count.get(model, 0)} skipped (missing runs or CSVs)"
        )
