"""Pool per-dataset result dicts of one configuration into a single combined result."""

import copy

METHOD_BLOCKS = ("pot", "pot_expanded", "oracle", "oracle_expanded")


def _count(block, name):
    """Read a confusion count from a block, accepting upper- or lower-case keys."""
    return float(block.get(name, block.get(name.lower(), 0)) or 0)


def _pooled_block(blocks):
    """Sum confusion counts over `blocks` and recompute the derived metrics.

    Keeps the key case of the first block (POT blocks use TP/FP/..., oracle
    blocks tp/fp/...). The threshold is taken from the first block (shared
    thresholds are identical across datasets); latency is the mean over blocks
    that report one.
    """
    first = blocks[0]
    upper = "TP" in first
    tp, fp, fn, tn = (sum(_count(b, k) for b in blocks) for k in ("TP", "FP", "FN", "TN"))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    latencies = [b["p_latency"] for b in blocks if b.get("p_latency") is not None]
    out = {
        "f1": f1, "precision": prec, "recall": rec, "fpr": fpr,
        "threshold": first.get("threshold"),
        "p_latency": sum(latencies) / len(latencies) if latencies else None,
    }
    counts = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    out.update(counts if upper else {k.lower(): int(v) for k, v in counts.items()})
    return out


def pooled_result(results, blocks_key=None):
    """Combine the per-dataset results of one model configuration into one result dict.

    With `blocks_key` (e.g. ``"shared"``), each result's method blocks are
    taken from ``result[blocks_key]`` when present (see ``shared.with_blocks``);
    results lacking it contribute their local blocks and the output's
    ``shared_threshold`` flag is False.

    Each method block (``pot``, ``pot_expanded``, ``oracle``,
    ``oracle_expanded``) gets pooled confusion counts and metrics recomputed
    from them. ``calibration_loss`` is the mean and ``eval_time`` the sum over
    datasets. ``model`` and ``applied_hyperparameters`` come from the first
    result; ``dataset`` becomes ``"combined"`` and ``datasets`` lists the
    sources. No ``_source_path`` is set, so per-run plots are skipped.
    """
    results = [r for r in results if r]
    if not results:
        return None
    if blocks_key:
        from AaltoAD.thresholds.shared import with_blocks
        results = [with_blocks(r, blocks_key) or r for r in results]
    first = results[0]
    out = {
        "model": first.get("model"),
        "applied_hyperparameters": copy.deepcopy(first.get("applied_hyperparameters", {})),
        "dataset": "combined",
        "datasets": [r.get("dataset") for r in results],
        "experiment_id": [r.get("experiment_id") for r in results],
        "shared_threshold": all(r.get("shared_threshold", False) for r in results),
    }
    for key, agg in (("calibration_loss", lambda v: sum(v) / len(v)), ("eval_time", sum)):
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            out[key] = agg(values)
    for name in METHOD_BLOCKS:
        blocks = [r[name] for r in results if isinstance(r.get(name), dict)]
        if len(blocks) == len(results):
            out[name] = _pooled_block(blocks)
    return out
