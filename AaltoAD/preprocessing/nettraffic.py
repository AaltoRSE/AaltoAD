# Preprocessing for data with the following columns:
# epoch_time,protocol,src_mac,dst_mac,src_ip,dst_ip,src_port,dst_port,framelen,has_temp,temperature

import os

import numpy as np
import pandas as pd

from AaltoAD.constants import DEFAULT_DATA_FOLDER
from AaltoAD.preprocessing import traffic_csv
from AaltoAD.preprocessing.utils import normalize

DEFAULT_SEGMENTS = [
    ("train", False, 300),
    ("calib", False, 60),
]

# Dataspec aggregations pandas has no name for. Every other aggregation is
# passed to pandas by name (count, sum, mean, nunique, ...).
CUSTOM_AGGREGATIONS = {"entropy": traffic_csv.shannon_entropy}


def _load_and_prepare(csv_path, interval=1, columns=None):
    """Read CSV, detect needed columns, parse timestamps. Returns a     parsed-CSV record.

    'columns' maps purpose -> CSV column name; each mapped column is renamed
    to its purpose before detection.
    """
    df = pd.read_csv(csv_path, on_bad_lines="warn")
    # Drop embedded header rows. These appear when several CSVs are concatenated
    # (e.g. `cat a.csv b.csv > combined.csv`), repeating the header line as data.
    # Left in, they force numeric columns to object dtype and skew the time range.
    header_row = df.columns.to_numpy().astype(str)
    header_mask = df.astype(str).eq(header_row).all(axis=1)
    if header_mask.any():
        print(
            f"  Dropping {int(header_mask.sum())} embedded header row(s) in {csv_path}"
        )
        df = df[~header_mask].reset_index(drop=True)
    if columns:
        missing = [name for name in columns.values() if name not in df.columns]
        if missing:
            raise ValueError(
                f"Mapped column(s) {missing} not found in {csv_path}; got columns {list(df.columns)}"
            )
        df = df.rename(columns={name: purpose for purpose, name in columns.items()})
    timestamp_col = traffic_csv.find_timestamp_column(df)
    bytes_col = traffic_csv.detect_column(
        df, ["bytes", "length", "len", "pkt_size", "size", "octets", "framelen"]
    )
    if bytes_col:
        df[bytes_col] = pd.to_numeric(df[bytes_col], errors="coerce").fillna(0)
    df["ts_sec"] = traffic_csv.parse_timestamp_column(df[timestamp_col])
    df["interval"] = df["ts_sec"] // interval
    df = df.dropna(subset=["interval"])
    df["interval"] = df["interval"].astype(int)
    df.drop(columns=["ts_sec"], inplace=True)
    df.drop(columns=[timestamp_col], inplace=True)
    temperature_col = traffic_csv.detect_column(df, ["temperature", "temp"])
    if temperature_col:
        df[temperature_col] = pd.to_numeric(df[temperature_col], errors="coerce")
    if "has_temp" in df.columns:
        has_temp_mask = (
            pd.to_numeric(df["has_temp"], errors="coerce").fillna(0).astype(bool)
        )
        df.loc[~has_temp_mask, temperature_col] = np.nan
    return {
        "path": csv_path,
        "df": df,
    }


def _cut_segments(parsed, segs, anomalies, interval):
    """Cut one parsed CSV into segment dicts.

    `segs` are (partition, anomalous, rows) with lengths in rows of `interval`
    seconds, laid back to back from the CSV's first interval; "skip" segments
    are dropped. `anomalies` are (start, duration) in seconds from the CSV start.
    Each segment dict has:
    - type: the partition (train, calib, test or valid)
    - source: the CSV path
    - start: start time in epoch seconds (aligned to `interval`)
    - duration: length in seconds
    - anomalies: (start, end) in seconds from the segment start, clipped to the
      segment; empty if no anomaly overlaps it
    - df: the raw rows that fall in the segment
    Raises ValueError if the segments run past the CSV's last interval.
    """
    df = parsed["df"]
    first = int(df["interval"].min())
    last = int(df["interval"].max())
    if first + sum(rows for _, _, rows in segs) > last + 1:
        raise ValueError(
            f"CSV {parsed['path']}: segments cover "
            f"{sum(rows for _, _, rows in segs) * interval}s but the data covers "
            f"{(last + 1 - first) * interval}s"
        )
    segments = []
    offset = 0
    for partition, _, rows in segs:
        if partition != "skip":
            seg_offset, duration = offset * interval, rows * interval
            ranges = []
            for a_start, a_duration in anomalies:
                start = max(a_start - seg_offset, 0)
                end = min(a_start + a_duration - seg_offset, duration)
                if start < end:
                    ranges.append((start, end))
            in_segment = (df["interval"] >= first + offset) & (
                df["interval"] < first + offset + rows
            )
            segments.append(
                {
                    "type": partition,
                    "source": parsed["path"],
                    "start": (first + offset) * interval,
                    "duration": duration,
                    "anomalies": ranges,
                    "df": df[in_segment],
                }
            )
        offset += rows
    return segments


def add_grouped_columns(segments, dataspec):
    """Add a '{col}_grouped' column for every group_by column in the dataspec.

    The values of all group_by columns are counted together over the rows of
    every train segment, the top `dataspec["group_by_count_limit"]` are kept,
    and every other value becomes "other" in the new columns. `segments` are
    segment dicts (see `_cut_segments`). Returns new segment dicts whose "df"
    has the new columns added (the input segments are not modified), and the
    values the new columns can hold: the kept values sorted, then "other".
    """
    group_by_cols = sorted({feature["group_by"] for feature in dataspec["features"]})

    baseline_values = [
        segment["df"][col].dropna().astype(str).str.strip()
        for segment in segments
        if segment["type"] == "train"
        for col in group_by_cols
    ]
    if not baseline_values:
        raise ValueError(
            "No train segments found across any CSV — cannot rank the group_by values"
        )
    all_values = pd.concat(baseline_values)
    all_values = all_values[all_values != ""]
    counts = all_values.value_counts()
    if counts.empty:
        raise ValueError(
            f"No values in group_by columns {group_by_cols} in train segments across any CSV"
        )
    kept_values = set(counts.head(dataspec["group_by_count_limit"]).index)
    print(
        f"nettraffic: {len(counts)} unique values in group_by columns {group_by_cols} "
        f"across all baselines; keeping the top {len(kept_values)}"
    )

    out = []
    for segment in segments:
        df = segment["df"]
        new_cols = {}
        for col in group_by_cols:
            values = df[col].astype(str).str.strip()
            new_cols[f"{col}_grouped"] = values.where(values.isin(kept_values), "other")
        out.append({**segment, "df": df.assign(**new_cols)})
    return out, sorted(kept_values) + ["other"]


def _aggregate_segment(segment, features, feature_cols, interval):
    """Aggregate one segment's raw rows into a feature table, one row per interval.

    For every dataspec feature the rows are grouped by interval and by the
    feature's '{group_by}_grouped' column, and its "column" is aggregated with
    its "aggregation" (column "any" counts rows, so its aggregation must be
    "count"). Columns are named '{value}_{feature name}'. The result has exactly
    the columns `feature_cols`, in that order, and one row per interval of the
    whole segment, `duration // interval` of them; anything without data is 0.
    """
    df = segment["df"]
    first = segment["start"] // interval
    index = pd.RangeIndex(
        first, first + segment["duration"] // interval, name="interval"
    )
    tables = []
    for feature in features:
        group = df.groupby(["interval", f"{feature['group_by']}_grouped"])
        if feature["column"] == "any":
            if feature["aggregation"] != "count":
                raise ValueError(
                    f"Feature {feature['name']!r}: column 'any' only supports aggregation 'count'"
                )
            values = group.size()
        else:
            aggregation = CUSTOM_AGGREGATIONS.get(
                feature["aggregation"], feature["aggregation"]
            )
            values = group[feature["column"]].agg(aggregation)
        table = values.unstack()
        table.columns = [f"{value}_{feature['name']}" for value in table.columns]
        tables.append(table)
    return (
        pd.concat(tables, axis=1)
        .reindex(index=index, columns=feature_cols)
        .fillna(0)
        .astype(float)
    )


def load_nettraffic(
    folder,
    csv_groups=None,
    csv_path=None,
    segments=None,
    data_folder=DEFAULT_DATA_FOLDER,
    columns=None,
    anomalies=None,
    dataspec=None,
):
    """Load and preprocess one or more CSV files containing network
    traffic data and temperature readings.

    The CSV file should contain individual records of network traffic,
    each row representing a single message or connection. The row may
    also contain a temperature reading.

    The files must have the following columns:
    - epoch_time: Unix timestamp of the record
    - src_ip: Source of the message
    - dst_ip: Destination of the message

    Optional columns may be used if present:
     - protocol: Network protocol used (e.g., TCP, UDP)
     - src_mac: Source MAC address
     - dst_mac: Destination MAC address
     - src_port: Source port number
     - dst_port: Destination port number
     - framelen: Length of the message in bytes
     - has_temp: Boolean indicating if a temperature reading is present
     - temperature: The temperature reading associated with the record

    'columns' maps purpose -> CSV column name (e.g. {"timestamp": "epoch_time"});
    each mapped column is renamed to its purpose after loading. A mapped column
    missing from a CSV raises ValueError.

    'anomalies' lists (start, duration) pairs in seconds from the start of each
    CSV, applied to every CSV. Overlapping test and valid rows are labeled
    anomalous.

    'dataspec' is the preprocessing definition loaded from the --dataspec JSON
    file (e.g. ELEC_dataspec.json), as a dict; None when not given. Its
    "interval" sets the aggregation interval in seconds. Feature names must be
    unique.

    Data is aggregated into time intervals specified by interval (in seconds). The features are counts of unique src/dst IPs for each top_K IP address, port entropy and mean temperature.
    """
    interval = dataspec["interval"]
    names = [feature["name"] for feature in dataspec["features"]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate feature names in the dataspec: {duplicates}")

    # Normalize input modes into a uniform csv_groups list.
    if csv_groups is None:
        if csv_path is None:
            csv_path = os.path.join(data_folder, "sample_data.csv")
        if segments is None:
            segments = DEFAULT_SEGMENTS
        csv_groups = [(csv_path, segments)]
    if not csv_groups:
        raise ValueError("At least one csv_paths and segments must be provided")

    # Calidate each group's segments.
    for paths, segs in csv_groups:
        if not paths:
            raise ValueError("Each group must have at least one csv path")
        if not segs:
            raise ValueError("Each group must have at least one segment")
        traffic_csv.validate_segments(segs)
        for _, _, seconds in segs:
            if seconds % interval:
                raise ValueError(
                    f"Segment length {seconds}s is not divisible by interval {interval}s"
                )

    # Convert segment lengths from seconds to aggregated rows.
    csv_groups = [
        (paths, [(p, a, seconds // interval) for p, a, seconds in segs])
        for paths, segs in csv_groups
    ]

    # Anomaly ranges must fall on interval boundaries.
    for start, duration in anomalies or []:
        if start % interval or duration % interval:
            raise ValueError(
                f"Anomaly {start}:{duration} is not divisible by interval {interval}s"
            )

    # Load each csv, parse columns and timestamp, and cut it into segment dicts.
    segments = []
    for paths, segs in csv_groups:
        for path in paths:
            parsed = _load_and_prepare(path, interval=interval, columns=columns)
            segments.extend(_cut_segments(parsed, segs, anomalies or [], interval))

    # Add the group_by columns with values outside the top ones set to "other".
    segments, grouped_values = add_grouped_columns(segments, dataspec)

    # The columns every segment gets, in this order.
    feature_cols = [
        f"{value}_{feature['name']}"
        for feature in dataspec["features"]
        for value in grouped_values
    ]

    # Aggregate each segment.
    for segment in segments:
        segment["features"] = _aggregate_segment(
            segment, dataspec["features"], feature_cols, interval
        )

    # Stack the features, labels and segment lengths of each partition.
    n_feat = len(feature_cols)
    data, partition_labels, lengths = {}, {}, {}
    for name in ("train", "calib", "test", "valid"):
        parts = [segment for segment in segments if segment["type"] == name]
        label_parts = []
        for segment in parts:
            segment_labels = np.zeros((len(segment["features"]), n_feat))
            for start, end in segment["anomalies"]:
                segment_labels[start // interval : end // interval] = 1
            label_parts.append(segment_labels)
        if parts:
            data[name] = np.concatenate(
                [segment["features"].values for segment in parts], axis=0
            )
            partition_labels[name] = np.concatenate(label_parts, axis=0)
        else:
            data[name] = np.empty((0, n_feat))
            partition_labels[name] = np.empty((0, n_feat))
        lengths[name] = [len(segment["features"]) for segment in parts]

    train, calib, test, valid = data["train"], data["calib"], data["test"], data["valid"]

    if train.size == 0:
        raise ValueError("Training partition is empty after split. Cannot normalize.")

    # Zero out columns that are constant in the training (baseline) data across
    # every partition. Without this, the eps-regularised divisor in normalize()
    # amplifies any calib/test variation in such columns by 1/eps (~10000x),
    # drowning out signal from the well-behaved features.
    keep = train.max(axis=0) != train.min(axis=0)
    if not keep.all():
        constant_cols = [c for c, k in zip(feature_cols, keep) if not k]
        print(
            f"  zeroing {len(constant_cols)} baseline-constant columns: {constant_cols}"
        )
        for part in (train, calib, test, valid):
            if part.size:
                part[:, ~keep] = 0.0

    train, min_a, max_a = normalize(train)
    if calib.size > 0:
        calib, _, _ = normalize(calib, min_a, max_a)
    if test.size > 0:
        test, _, _ = normalize(test, min_a, max_a)
    if valid.size > 0:
        valid, _, _ = normalize(valid, min_a, max_a)

    labels = partition_labels["test"]
    valid_labels = partition_labels["valid"]
    print(
        f"   Anomaly labels: {int(labels.sum() / n_feat) if labels.size else 0}/{labels.shape[0]} test rows, "
        f"{int(valid_labels.sum() / n_feat) if valid_labels.size else 0}/{valid_labels.shape[0]} valid rows"
    )

    for name, arr in [
        ("train", train),
        ("test", test),
        ("calib", calib),
        ("valid", valid),
        ("labels", labels),
        ("valid_labels", valid_labels),
    ]:
        np.save(
            os.path.join(folder, f"{name}.npy"),
            np.ascontiguousarray(arr.astype("float64")),
        )
    # Row count of each segment, to split the concatenated partition arrays.
    for name, segment_lengths in lengths.items():
        np.save(
            os.path.join(folder, f"{name}_lengths.npy"),
            np.array(segment_lengths, dtype=int),
        )
    all_paths = [p for paths, _ in csv_groups for p in paths]
    print(f"Processed {len(all_paths)} CSV(s) as nettraffic -> {folder}/")
    print(
        f"  train.npy: {train.shape}, test.npy: {test.shape}, calib.npy: {calib.shape}, valid.npy: {valid.shape}, labels.npy: {labels.shape}"
    )
