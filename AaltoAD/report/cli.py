"""CLI option strings, defaults, and input normalization for dataset reports."""

from AaltoAD import constants


# Option strings for the CLI
METHODS = ["pot", "oracle", "conformal"]

METHOD_METRICS = ["f1", "precision", "recall", "fpr", "threshold", "p_latency"]

METRIC_ALIASES = {"latency": "p_latency", "precision": "precision", "recall": "recall"}

# Suffix marking the point-adjusted variant of a method block ("pot_expanded"),
# reached by dotted paths like "{method}_expanded.f1".
EXPANDED_SUFFIX = "_expanded"

# Whether one threshold per configuration is fit on the calibration data of all
# datasets pooled together (--pool-baselines), instead of one per dataset.
POOL_BASELINES = constants.POOL_BASELINES

# Default threshold method the report quotes and scales by (--threshold-method
# overrides). Table headers carry no method label, so every generated table and
# plot in one report refers to one and the same method.
THRESHOLD_METHOD = constants.THRESHOLD_METHOD

# Header and lookup for every column --table-columns can name. An "_expanded"
# path reaches the point-adjusted block of the same method.
TABLE_COLUMN_SPECS = {
    "f1": ("F1", "{method}.f1"),
    "adjusted_f1": ("Adjusted F1", "{method}_expanded.f1"),
    "precision": ("Precision", "{method}.precision"),
    "recall": ("Recall", "{method}.recall"),
    "fpr": ("FPR", "{method}.fpr"),
    "p_latency": ("Latency", "{method}.p_latency"),
    "threshold": ("Threshold", "{method}.threshold"),
    "calibration_loss": ("Calib. loss", "calibration_loss"),
    "eval_time": ("Eval time (s)", "eval_time"),
}

TABLE_COLUMNS = tuple(constants.TABLE_COLUMNS)

TABLE_BLOCKS = constants.TABLE_BLOCKS


def _split_cli_list(value):
    """Split a comma/space-separated CLI value into stripped, non-empty parts."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(" ", ",").split(",")]
    else:
        parts = [str(p).strip() for p in value]
    return [p for p in parts if p]


def metric_list(metric):
    """Normalize a `--metric` argument into a tuple of bare metric names.

    Accepts a comma-separated string, a sequence, or a single name, and takes
    the metric out of a dotted path, so 'latency,fpr,f1', ('p_latency', 'fpr',
    'f1') and 'conformal.f1' all work. The threshold method comes from
    `--threshold`, not from here.
    """
    names = []
    for part in _split_cli_list(metric):
        name = part.split(".")[-1] if "." in part else part
        names.append(METRIC_ALIASES.get(name, name))
    return tuple(names)


def metric_display(metric):
    """Human-readable form of a metric argument, for table headers and titles."""
    return ",".join(metric_list(metric))


def method_name(value):
    """Method named by `value`: 'conformal' and 'conformal.f1' both give 'conformal'."""
    name = str(value).split(".")[0]
    return name[: -len(EXPANDED_SUFFIX)] if name.endswith(EXPANDED_SUFFIX) else name


def table_column_list(columns):
    """Normalize a `--table-columns` argument into a tuple of column names."""
    return tuple(METRIC_ALIASES.get(p, p) for p in _split_cli_list(columns))
