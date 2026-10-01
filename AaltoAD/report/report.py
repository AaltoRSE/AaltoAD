"""Generate dataset reports (HTML, PDF, CSV, LaTeX) from sweep result JSON files,
selecting and ordering each model's runs by metric."""

import json
import os

from glob import glob
from tqdm import tqdm

from AaltoAD import constants
from AaltoAD.report import cli, metrics, plot, tables
from AaltoAD.thresholds import shared


def _load_results(dataset, results_folder="results"):
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


def _apply_shared_thresholds(by_dataset, results_folder, conformal_q=constants.CONFORMAL_Q,
                             pool_baselines=cli.POOL_BASELINES):
    """Attach conformal/pot/oracle blocks to every result, in place.

    With `pool_baselines`, one threshold per configuration is fit on the
    calibration data of every dataset pooled together — which assumes the
    baselines are alike, and lets a contaminated one set the threshold for all
    the rest. By default each dataset is fit on its own baseline instead, by
    running the same machinery once per dataset; the blocks still land where
    the pooled ones would, so the combined report and the shared plots keep
    working on separately fitted thresholds.
    """
    if not pool_baselines and len(by_dataset) > 1:
        for dataset, by_model in by_dataset.items():
            _fit_threshold_blocks({dataset: by_model}, results_folder, conformal_q)
        return
    _fit_threshold_blocks(by_dataset, results_folder, conformal_q)


def _fit_threshold_blocks(by_dataset, results_folder, conformal_q=constants.CONFORMAL_Q):
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
    which is how `_apply_shared_thresholds` fits separate baselines.
    """

    datasets = list(by_dataset)
    groups = shared.group_configurations(
        by_dataset, metrics._hp_key, lambda rs: metrics._best_result(rs, "calibration_loss"))
    complete = {k: v for k, v in groups.items() if set(v) == set(datasets)}
    print(f"Fitting thresholds for {len(complete)} configurations on the calibration of {datasets}")

    cache_file = shared.cache_path(results_folder, datasets)
    cache = shared.load_cache(cache_file)
    shared_count, skipped_count = {}, {}
    for (model, _), members in groups.items():
        if set(members) != set(datasets):
            skipped_count[model] = skipped_count.get(model, 0) + 1

    for (model, hp_key), results_by_dataset in tqdm(complete.items(), desc="shared thresholds", unit="config"):
        q = next(iter(results_by_dataset.values())).get("applied_hyperparameters", {}).get("q", 1e-5)
        constants.initialize(datasets[0], model)
        blocks = shared.shared_blocks_cached(cache, model, hp_key, results_by_dataset, q,
                                             constants.level, conformal_q)
        if blocks is None:
            skipped_count[model] = skipped_count.get(model, 0) + 1
            continue
        shared.apply_blocks(results_by_dataset, blocks)
        shared_count[model] = shared_count.get(model, 0) + 1

    shared.save_cache(cache_file, cache)
    for model in sorted(set(shared_count) | set(skipped_count)):
        print(f"  {model}: {shared_count.get(model, 0)} configuration(s) shared, "
              f"{skipped_count.get(model, 0)} skipped (missing runs or CSVs)")


def generate_report(args, results_folder="results"):
    """Generate HTML, PDF, CSV, and hyperparameter-markdown reports (--report mode).

    `args` is the namespace parsed by `AaltoAD.parser`; `results_folder` is
    where the per-run result JSONs live.

    `args.dataset` may be a single name, a comma-separated string, or a list of
    names. With several datasets, one hyperparameter set per model is selected
    by the best *sum* of `args.metric` over all of them, and each dataset's
    report shows that shared configuration with its local (per-dataset)
    thresholds. POT and oracle thresholds are also fit jointly across the
    datasets (see `_apply_shared_thresholds`); those shared thresholds drive
    the reports/combined/ summary (confusion counts pooled over the datasets)
    and a second set of plots per dataset. Files are saved in
    reports/{dataset}/. `args.plot_models` is how many models each
    prediction-error overlay shows; the models are the best `args.plot_models`
    by `args.metric` on that dataset — unless a hand-picked
    ``reports/<dataset>/model-order.json`` exists (a JSON array of model names;
    the combined report reads ``reports/combined/model-order.json``), in which
    case the overlay shows exactly those models in that order, a name with no
    results skipped with a warning; tables and orderings are unaffected by the
    file (see `plot._plot_model_selection`). A multi-dataset run also writes the
    cross-case overview tables to reports/overview_*.tex (see
    `_generate_overview_latex`).

    `args.threshold` names the threshold method every table quotes and every
    plot is scaled by (normalized with `cli.method_name`). `args.metric` is a
    comma-separated list of metric names inside it (see `cli.metric_list`): it
    selects each model's run — aggregated over the datasets, an F1 recomputed
    from pooled counts rather than averaged — and orders the models everywhere,
    unless `args.model_order` gives a separate ordering. `args.conformal_q` is
    the false alarm rate the conformal threshold targets, `args.pool_baselines`
    whether one threshold is fit across all the datasets' calibration data
    instead of one per dataset, and `args.downsample`/`args.downsample_window`
    decide how plotted series are reduced, `args.table_columns` which columns
    the LaTeX summary shows and `args.table_blocks` how many models it sets
    side by side.
    """
    # One metric does both jobs unless the caller separates them.
    metric = cli.metric_list(args.metric)
    model_order = metric if args.model_order is None else cli.metric_list(args.model_order)
    threshold_method = cli.method_name(args.threshold)
    n_plot_models = args.plot_models
    conformal_q = args.conformal_q
    pool_baselines = args.pool_baselines
    downsample_mode = args.downsample
    downsample_window = args.downsample_window
    table_columns = args.table_columns
    table_blocks = args.table_blocks

    if isinstance(args.dataset, str):
        datasets = [d for d in args.dataset.split(",") if d]
    else:
        datasets = list(args.dataset)

    by_dataset = {}
    for ds in datasets:
        by_model = _load_results(ds, results_folder)
        if not by_model:
            print(f'No results found for dataset "{ds}" in {results_folder}/')
            continue
        by_dataset[ds] = by_model
    if not by_dataset:
        return

    # Always fit the shared thresholds: the conformal threshold is defined by
    # the calibration sample it is fitted on, and this is where that fit happens
    # (with one dataset listed, "shared across the datasets" is just that one).
    _apply_shared_thresholds(by_dataset, results_folder, conformal_q, pool_baselines)

    selected = metrics._select_shared_best(by_dataset, metric, threshold_method)
    # Each dataset's overlay picks its own best `n_plot_models` (plot_models=None),
    # so a model that only wins on one case still shows up there — unless the
    # dataset's model-order.json hand-picks the plotted models (see
    # `plot._plot_model_selection`). The combined report is a single pooled
    # ranking, so it gets the pooled top list, or its own hand-picked file.
    for ds in by_dataset:
        _generate_dataset_report(ds, metric, selected[ds],
                                 plot_models=plot._plot_model_selection(ds, selected[ds]),
                                 shared_plots=len(by_dataset) > 1, n_plot_models=n_plot_models,
                                 threshold_method=threshold_method, model_order=model_order,
                                 downsample_mode=downsample_mode, downsample_window=downsample_window,
                                 table_columns=table_columns, table_blocks=table_blocks)
    if len(by_dataset) > 1:
        tables._generate_overview_latex(selected, metric, "reports", threshold_method=threshold_method,
                                 model_order=model_order)
        pooled = _pooled_by_model(selected)
        plot_models = plot._plot_model_selection("combined", pooled)
        if plot_models is None:
            plot_models = _top_models(pooled, metric, n=n_plot_models,
                                      threshold_method=threshold_method,
                                      model_order=model_order)
        _generate_dataset_report("combined", metric, pooled, plot_models=plot_models,
                                 n_plot_models=n_plot_models, threshold_method=threshold_method,
                                 model_order=model_order, downsample_mode=downsample_mode,
                                 downsample_window=downsample_window, table_columns=table_columns,
                                 table_blocks=table_blocks)


def _order_models(by_model, metric, threshold_method=cli.THRESHOLD_METHOD, model_order=None):
    """Model names in the report's display order, best first.

    `metric` selects each model's best run (its hyperparameters); the ranking
    across models reads `model_order` from the `threshold_method` block, or
    `metric` itself when no separate ordering is given (see `metrics._rank_key`). Every
    table and plot in a report uses this one order.
    """
    model_order = cli.metric_list(metric if model_order is None else model_order)
    return sorted(
        by_model,
        key=lambda m: metrics._rank_key(metrics._best_result(by_model[m], metric, threshold_method), model_order, threshold_method),
    )


def _order_rank(by_model, metric, threshold_method=cli.THRESHOLD_METHOD, model_order=None):
    """``{model: position}`` in the display order, for sorting already-built table rows."""
    return {m: i for i, m in enumerate(_order_models(by_model, metric, threshold_method, model_order))}


def _top_models(by_model, metric, n=plot.DEFAULT_PLOT_MODELS, threshold_method=cli.THRESHOLD_METHOD,
                model_order=None):
    """Names of the `n` models a plot should show, in `_order_models` order."""
    return _order_models(by_model, metric, threshold_method, model_order)[:n]


def _pooled_by_model(selected):
    """Pool each model's selected per-dataset results into one combined result, using the shared-threshold blocks."""
    models = {m for by_model in selected.values() for m in by_model}
    pooled = {}
    for model in sorted(models):
        combined = pooled.pooled_result([r for by_model in selected.values() for r in by_model.get(model, [])], blocks_key="shared")
        if combined:
            pooled[model] = [combined]
    return pooled


def _generate_dataset_report(dataset, metric, by_model, plot_models=None, shared_plots=False,
                             n_plot_models=plot.DEFAULT_PLOT_MODELS, threshold_method=cli.THRESHOLD_METHOD,
                             model_order=None, downsample_mode=plot.DOWNSAMPLE, downsample_window=None,
                             table_columns=cli.TABLE_COLUMNS, table_blocks=cli.TABLE_BLOCKS):
    """Write all report files for one dataset from its (pre-selected) results.

    `plot_models` fixes the models shown in the overlay plot (see
    `plot._generate_prediction_error_plot`). With `shared_plots`, a second set of
    plots scaled by the shared thresholds is written next to the local ones
    (``prediction_errors_shared.png`` and ``plots_shared/``).
    """
    # Determine unlabeled flag: True when at least one result has a pot confusion
    # matrix AND all results that have one satisfy metrics._is_unlabeled.
    results_with_pot = [
        r
        for results in by_model.values()
        for r in results
        if isinstance(r.get("pot"), dict) and r["pot"].get("TP") is not None
    ]
    unlabeled = bool(results_with_pot) and all(
        metrics._is_unlabeled(r) for r in results_with_pot
    )

    dataset_dir = os.path.join("reports", dataset)

    os.makedirs(dataset_dir, exist_ok=True)
    html_path = os.path.join(dataset_dir, f"report.html")
    pdf_path = os.path.join(dataset_dir, f"report.pdf")
    csv_path = os.path.join(dataset_dir, f"summary.csv")
    hp_path = os.path.join(dataset_dir, f"hyperparams.md")
    tex_path = os.path.join(dataset_dir, f"summary.tex")
    plot_path = os.path.join(dataset_dir, f"prediction_errors.png")
    plots_dir = os.path.join(dataset_dir, "plots")

    # One model order for every table and plot of this dataset. Unlabeled runs
    # have no detections to rank, so they keep their own metric sort.
    order = None if unlabeled else _order_rank(by_model, metric, threshold_method, model_order)

    tables._generate_html(dataset, metric, by_model, html_path, unlabeled=unlabeled, order=order)
    tables._generate_pdf(dataset, metric, by_model, pdf_path, unlabeled=unlabeled, order=order)
    tables._generate_csv(dataset, metric, by_model, csv_path, unlabeled=unlabeled, order=order)
    tables._generate_hp_markdown(dataset, metric, by_model, hp_path, order=order)
    tables._generate_latex(dataset, metric, by_model, tex_path, unlabeled=unlabeled,
                    threshold_method=threshold_method, order=order, columns=table_columns,
                    blocks=table_blocks)
    plot._generate_prediction_error_plot(dataset, metric, by_model, plot_path, models=plot_models,
                                    n_models=n_plot_models, threshold_method=threshold_method,
                                    model_order=model_order, downsample_mode=downsample_mode,
                                    downsample_window=downsample_window)
    plot._generate_model_plots(dataset, metric, by_model, plots_dir, threshold_method=threshold_method,
                          downsample_mode=downsample_mode, downsample_window=downsample_window)
    if shared_plots:
        shared_plot_path = os.path.join(dataset_dir, "prediction_errors_shared.png")
        plot._generate_prediction_error_plot(dataset, metric, by_model, shared_plot_path, models=plot_models,
                                        blocks_key="shared", n_models=n_plot_models,
                                        threshold_method=threshold_method, model_order=model_order,
                                        downsample_mode=downsample_mode,
                                        downsample_window=downsample_window)
        plot._generate_model_plots(dataset, metric, by_model, os.path.join(dataset_dir, "plots_shared"),
                              blocks_key="shared", threshold_method=threshold_method,
                              downsample_mode=downsample_mode, downsample_window=downsample_window)
