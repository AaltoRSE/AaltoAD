"""Generate dataset reports (HTML, PDF, CSV, LaTeX) from sweep result JSON files,
selecting and ordering each model's runs by metric."""

import os

from AaltoAD.report import cli, load_results, metrics, plot, tables


def generate_report(args, results_folder="results"):
    """Generate HTML, PDF, CSV, and hyperparameter-markdown reports (--report mode).

    `args` is the namespace parsed by `AaltoAD.parser`; `results_folder` is
    where the per-run result JSONs live.

    `args.dataset` may be a single name, a comma-separated string, or a list of
    names. With several datasets, one hyperparameter set per model is selected
    by the best *sum* of `args.metric` over all of them, and each dataset's
    report shows that shared configuration with its local (per-dataset)
    thresholds. POT and oracle thresholds are also fit jointly across the
    datasets (fitted in `load_results.load_results`); those shared thresholds drive
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
    model_order = (
        metric if args.model_order is None else cli.metric_list(args.model_order)
    )
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

    by_dataset = load_results.load_results(
        datasets, results_folder, conformal_q, pool_baselines
    )
    if not by_dataset:
        return

    selected = metrics._select_shared_best(by_dataset, metric, threshold_method)
    # Each dataset's overlay picks its own best `n_plot_models` (plot_models=None),
    # so a model that only wins on one case still shows up there — unless the
    # dataset's model-order.json hand-picks the plotted models (see
    # `plot._plot_model_selection`). The combined report is a single pooled
    # ranking, so it gets the pooled top list, or its own hand-picked file.
    for ds in by_dataset:
        _generate_dataset_report(
            ds,
            metric,
            selected[ds],
            plot_models=plot._plot_model_selection(ds, selected[ds]),
            shared_plots=len(by_dataset) > 1,
            n_plot_models=n_plot_models,
            threshold_method=threshold_method,
            model_order=model_order,
            downsample_mode=downsample_mode,
            downsample_window=downsample_window,
            table_columns=table_columns,
            table_blocks=table_blocks,
        )
    if len(by_dataset) > 1:
        tables._generate_overview_latex(
            selected,
            metric,
            "reports",
            threshold_method=threshold_method,
            model_order=model_order,
        )
        pooled = _pooled_by_model(selected)
        plot_models = plot._plot_model_selection("combined", pooled)
        if plot_models is None:
            plot_models = _top_models(
                pooled,
                metric,
                n=n_plot_models,
                threshold_method=threshold_method,
                model_order=model_order,
            )
        _generate_dataset_report(
            "combined",
            metric,
            pooled,
            plot_models=plot_models,
            n_plot_models=n_plot_models,
            threshold_method=threshold_method,
            model_order=model_order,
            downsample_mode=downsample_mode,
            downsample_window=downsample_window,
            table_columns=table_columns,
            table_blocks=table_blocks,
        )


def _order_models(
    by_model, metric, threshold_method=cli.THRESHOLD_METHOD, model_order=None
):
    """Model names in the report's display order, best first.

    `metric` selects each model's best run (its hyperparameters); the ranking
    across models reads `model_order` from the `threshold_method` block, or
    `metric` itself when no separate ordering is given (see `metrics._rank_key`). Every
    table and plot in a report uses this one order.
    """
    model_order = cli.metric_list(metric if model_order is None else model_order)
    return sorted(
        by_model,
        key=lambda m: metrics._rank_key(
            metrics._best_result(by_model[m], metric, threshold_method),
            model_order,
            threshold_method,
        ),
    )


def _order_rank(
    by_model, metric, threshold_method=cli.THRESHOLD_METHOD, model_order=None
):
    """``{model: position}`` in the display order, for sorting already-built table rows."""
    return {
        m: i
        for i, m in enumerate(
            _order_models(by_model, metric, threshold_method, model_order)
        )
    }


def _top_models(
    by_model,
    metric,
    n=plot.DEFAULT_PLOT_MODELS,
    threshold_method=cli.THRESHOLD_METHOD,
    model_order=None,
):
    """Names of the `n` models a plot should show, in `_order_models` order."""
    return _order_models(by_model, metric, threshold_method, model_order)[:n]


def _pooled_by_model(selected):
    """Pool each model's selected per-dataset results into one combined result, using the shared-threshold blocks."""
    models = {m for by_model in selected.values() for m in by_model}
    pooled = {}
    for model in sorted(models):
        combined = pooled.pooled_result(
            [r for by_model in selected.values() for r in by_model.get(model, [])],
            blocks_key="shared",
        )
        if combined:
            pooled[model] = [combined]
    return pooled


def _generate_dataset_report(
    dataset,
    metric,
    by_model,
    plot_models=None,
    shared_plots=False,
    n_plot_models=plot.DEFAULT_PLOT_MODELS,
    threshold_method=cli.THRESHOLD_METHOD,
    model_order=None,
    downsample_mode=plot.DOWNSAMPLE,
    downsample_window=None,
    table_columns=cli.TABLE_COLUMNS,
    table_blocks=cli.TABLE_BLOCKS,
):
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
    order = (
        None
        if unlabeled
        else _order_rank(by_model, metric, threshold_method, model_order)
    )

    tables._generate_html(
        dataset, metric, by_model, html_path, unlabeled=unlabeled, order=order
    )
    tables._generate_pdf(
        dataset, metric, by_model, pdf_path, unlabeled=unlabeled, order=order
    )
    tables._generate_csv(
        dataset, metric, by_model, csv_path, unlabeled=unlabeled, order=order
    )
    tables._generate_hp_markdown(dataset, metric, by_model, hp_path, order=order)
    tables._generate_latex(
        dataset,
        metric,
        by_model,
        tex_path,
        unlabeled=unlabeled,
        threshold_method=threshold_method,
        order=order,
        columns=table_columns,
        blocks=table_blocks,
    )
    plot._generate_prediction_error_plot(
        dataset,
        metric,
        by_model,
        plot_path,
        models=plot_models,
        n_models=n_plot_models,
        threshold_method=threshold_method,
        model_order=model_order,
        downsample_mode=downsample_mode,
        downsample_window=downsample_window,
    )
    plot._generate_model_plots(
        dataset,
        metric,
        by_model,
        plots_dir,
        threshold_method=threshold_method,
        downsample_mode=downsample_mode,
        downsample_window=downsample_window,
    )
    if shared_plots:
        shared_plot_path = os.path.join(dataset_dir, "prediction_errors_shared.png")
        plot._generate_prediction_error_plot(
            dataset,
            metric,
            by_model,
            shared_plot_path,
            models=plot_models,
            blocks_key="shared",
            n_models=n_plot_models,
            threshold_method=threshold_method,
            model_order=model_order,
            downsample_mode=downsample_mode,
            downsample_window=downsample_window,
        )
        plot._generate_model_plots(
            dataset,
            metric,
            by_model,
            os.path.join(dataset_dir, "plots_shared"),
            blocks_key="shared",
            threshold_method=threshold_method,
            downsample_mode=downsample_mode,
            downsample_window=downsample_window,
        )
