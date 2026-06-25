# Preprocessing for data with the following columns:
# epoch_time,protocol,src_mac,dst_mac,src_ip,dst_ip,src_port,dst_port,framelen,has_temp,temperature

import os

import numpy as np
import pandas as pd

from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import normalize

from .tol import (
    _baseline_mask,
    _canonical_columns,
    _detect_column,
    _detect_ip_columns,
    _shannon_entropy,
    _validate_segments,
    find_timestamp_column,
    parse_timestamp_column,
)

DEFAULT_SEGMENTS = [
    ("train", False, 300),
    ("calib", False, 60),
]


def _load_and_prepare(csv_path, interval=1):
    """Read CSV, detect needed columns, parse timestamps. Returns a     parsed-CSV record."""
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
    timestamp_col = find_timestamp_column(df)
    src_col, dst_col = _detect_ip_columns(df)
    bytes_col = _detect_column(
        df, ["bytes", "length", "len", "pkt_size", "size", "octets", "framelen"]
    )
    if bytes_col:
        df[bytes_col] = pd.to_numeric(df[bytes_col], errors="coerce").fillna(0)
    src_port_col = _detect_column(df, ["src_port", "sport", "source_port"])
    df["ts_sec"] = parse_timestamp_column(df[timestamp_col])
    df["interval"] = df["ts_sec"] // interval
    df = df.dropna(subset=["interval"])
    df["interval"] = df["interval"].astype(int)
    df.drop(columns=["ts_sec"], inplace=True)
    df.drop(columns=[timestamp_col], inplace=True)
    temperature_col = _detect_column(df, ["temperature", "temp"])
    if temperature_col:
        df[temperature_col] = pd.to_numeric(df[temperature_col], errors="coerce")
    print(
        f"  Detected columns in {csv_path}: timestamp='{timestamp_col}', src_ip='{src_col}', dst_ip='{dst_col}', bytes='{bytes_col}', src_port='{src_port_col}', temp='{temperature_col}'"
    )
    if "has_temp" in df.columns:
        has_temp_mask = (
            pd.to_numeric(df["has_temp"], errors="coerce").fillna(0).astype(bool)
        )
        df.loc[~has_temp_mask, temperature_col] = np.nan
    return {
        "path": csv_path,
        "df": df,
        "src_col": src_col,
        "dst_col": dst_col,
        "bytes_col": bytes_col,
        "port_col": src_port_col,
        "temp_col": temperature_col,
    }


def _canonical_columns(top_ips, has_bytes, has_ports, has_temp, include_other):
    """Build the canonical column list shared across all CSVs in a combined dataset."""
    ip_keys = sorted(top_ips)
    if include_other:
        ip_keys += ["other_internal", "other_external"]
    cols = []
    # Outgoing block (per-IP num_dsts + extended)
    for ip in ip_keys:
        cols.append(f"{ip}_num_dsts")
    for ip in ip_keys:
        cols.append(f"{ip}_out_count")
    if has_bytes:
        for ip in ip_keys:
            cols.append(f"{ip}_bytes_out")
    if has_ports:
        for ip in ip_keys:
            cols.append(f"{ip}_port_entropy")
    if has_temp:
        for ip in ip_keys:
            cols.append(f"{ip}_mean_temp")
    # Incoming block (per-IP num_srcs + extended)
    for ip in ip_keys:
        cols.append(f"{ip}_num_srcs")
    for ip in ip_keys:
        cols.append(f"{ip}_in_count")
    if has_bytes:
        for ip in ip_keys:
            cols.append(f"{ip}_bytes_in")
        # Derived
        for ip in ip_keys:
            cols.append(f"{ip}_rw_ratio")
    # Global
    cols.append("rows_per_interval")
    cols.append("empty_rows")
    return cols


def _aggregate_to_features(
    parsed, top_ips, internal_prefix, port_entropy=False, bytes=False, temperature=False
):
    """Aggregate one parsed CSV into per second features dataframe.

    Uses the supplied 'top_ips' and 'internal_prefix' to map the columm
    space accross CSVs that share a baseline. Non-top-k IPS are grouped
    into 'other_internal' vs 'other_external' based on the internal_prefix.
    """
    df = parsed["df"].copy()
    src_col, dst_col = parsed["src_col"], parsed["dst_col"]
    bytes_col, port_col, temp_col = (
        parsed["bytes_col"],
        parsed["port_col"],
        parsed["temp_col"],
    )

    def _classify_ip(ip):
        if internal_prefix is None:
            return str(ip).strip()
        ip_str = str(ip).strip()
        if ip_str in top_ips:
            return ip_str
        return (
            "other_internal" if ip_str.startswith(internal_prefix) else "other_external"
        )

    df[src_col] = df[src_col].astype(str).str.strip().map(_classify_ip)
    df[dst_col] = df[dst_col].astype(str).str.strip().map(_classify_ip)

    rows_per_interval = None
    empty_rows = None
    rows_per_interval = df.groupby("interval").size().rename("rows_per_interval")
    empty_rows = (df[src_col].isna() | (df[src_col].astype(str).str.strip() == "")) & (
        df[dst_col].isna() | (df[dst_col].astype(str).str.strip() == "")
    )
    empty_rows = df[empty_rows].groupby("interval").size().rename("empty_rows")

    out_grp = df.groupby(["interval", src_col])
    out_agg = pd.DataFrame({"num_dsts": out_grp[dst_col].nunique()})
    out_agg["out_count"] = out_grp.size()
    if bytes and bytes_col:
        out_agg["bytes_out"] = out_grp[bytes_col].sum()
    if port_entropy and port_col:
        out_agg["port_entropy"] = out_grp[port_col].apply(_shannon_entropy)
    if temperature and temp_col:
        out_agg["mean_temp"] = out_grp[temp_col].mean()

    in_grp = df.groupby(["interval", dst_col])
    in_agg = pd.DataFrame({"num_srcs": in_grp[src_col].nunique()})
    in_agg["in_count"] = in_grp.size()
    if bytes and bytes_col:
        in_agg["bytes_in"] = in_grp[bytes_col].sum()

    out_wide = out_agg.unstack(fill_value=0)
    out_wide.columns = [f"{ip}_{metric}" for metric, ip in out_wide.columns]
    in_wide = in_agg.unstack(fill_value=0)
    in_wide.columns = [f"{ip}_{metric}" for metric, ip in in_wide.columns]

    ts_min, ts_max = df["interval"].min(), df["interval"].max()
    full_idx = np.arange(ts_min, ts_max + 1)
    features_df = (
        out_wide.reindex(full_idx, fill_value=0)
        .join(in_wide.reindex(full_idx, fill_value=0), how="outer")
        .fillna(0)
    )
    features_df = (
        features_df.join(rows_per_interval.reindex(full_idx, fill_value=0), how="outer")
        .join(empty_rows.reindex(full_idx, fill_value=0), how="outer")
        .fillna(0)
    )
    features_df.index.name = "interval"

    feature_list = list(top_ips)
    if internal_prefix is not None:
        feature_list += ["other_internal", "other_external"]

    for ip in feature_list:
        b_in_col = f"{ip}_bytes_in"
        b_out_col = f"{ip}_bytes_out"
        if b_in_col in features_df.columns and b_out_col in features_df.columns:
            features_df[f"{ip}_rw_ratio"] = features_df[b_out_col] / (
                features_df[b_in_col] + 1e-6
            )

    return features_df


def load_nettraffic(
    folder,
    csv_groups=None,
    csv_path=None,
    segments=None,
    data_folder=DEFAULT_DATA_FOLDER,
    top_k=10,
    interval=1,
    bytes=False,
    port_entropy=False,
    temperature=False,
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


    Data is aggregated into time intervals specified by interval (in seconds). The features are counts of unique src/dst IPs for each top_K IP address, port entropy and mean temperature.
    """

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
        _validate_segments(segs)

    # Load each csv, parse columns and timestamp. Collect baseline IPs.
    parsed_records = []
    all_baseline_ips = []
    for paths, segs in csv_groups:
        for path in paths:
            parsed = _load_and_prepare(path, interval=interval)
            parsed_records.append((parsed, segs))
            mask = _baseline_mask(parsed["df"], segs, interval_col="interval")
            if mask.any():
                all_baseline_ips.append(
                    parsed["df"]
                    .loc[mask, parsed["src_col"]]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                all_baseline_ips.append(
                    parsed["df"]
                    .loc[mask, parsed["dst_col"]]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

    if not all_baseline_ips:
        raise ValueError(
            "No train segments found across any CSV — cannot determine baseline IP set"
        )

    all_ips_concat = pd.concat(all_baseline_ips)
    all_ips_concat = all_ips_concat[all_ips_concat != ""]
    counts = all_ips_concat.value_counts()
    if counts.empty:
        raise ValueError("No valid IP addresses found in train segments across any CSV")

    most_common_ips = counts.idxmax()
    top_ips = set(counts.head(top_k).index)

    if len(top_ips) < top_k:
        print("Using all IPs from baseline since unique IP count is less than top_k")
        internal_prefix = None
        use_all_ips = True
    else:
        internal_prefix = most_common_ips.split(".")[0]
        use_all_ips = False

    total_baseline_rows = (
        sum(len(s) for s in all_baseline_ips) // 2
    )  # we appended src + dst
    print(
        f"TOL: unique IPs across all baselines: {len(counts)} "
        f"(total baseline rows aggregated: {total_baseline_rows})"
    )
    if not use_all_ips:
        print(
            f"TOL: most common IP {most_common_ips} => internal prefix {internal_prefix}"
        )
        print(
            f"TOL: selecting top_{top_k} IPs from combined baseline (keeps {len(top_ips)})"
        )

    # Determine canonical column names.
    has_bytes = bytes and any(
        "framelen" in parsed["df"].columns for parsed, _ in parsed_records
    )
    has_temp = temperature and any(
        "temperature" in parsed["df"].columns for parsed, _ in parsed_records
    )
    has_ports = port_entropy and any(
        "src_port" in parsed["df"].columns and "dst_port" in parsed["df"].columns
        for parsed, _ in parsed_records
    )
    canonical_cols = _canonical_columns(
        top_ips, has_bytes, has_ports, has_temp, not use_all_ips
    )

    # Aggregate each csv reindex to canonical columns, slice and accumulate
    partitions = {"train": [], "calib": [], "test": [], "valid": []}
    label_seqs = {"train": [], "calib": [], "test": [], "valid": []}
    for parsed, segs in parsed_records:
        features_df = _aggregate_to_features(
            parsed,
            top_ips,
            internal_prefix,
            port_entropy=port_entropy,
            bytes=bytes,
            temperature=temperature,
        )
        print(f"  CSV {parsed['path']} aggregated to {features_df.shape[0]} intervals")

        features_df = features_df.reindex(columns=canonical_cols, fill_value=0)
        features = features_df.values.astype(float)
        total_needed = sum(s[2] for s in segs[:-1])
        if features.shape[0] < total_needed:
            raise ValueError(
                f"CSV {parsed['path']}: segments require {total_needed} rows but only {features.shape[0]} available after aggregation"
            )
        if features.shape[0] < total_needed + segs[-1][2]:
            print(
                f"  Warning: CSV {parsed['path']} has fewer rows than total segments. Last segment will be truncated to fit available data."
            )
        offset = 0
        for partition, anomalous, seconds in segs:
            if partition == "skip":
                offset += seconds
                continue
            partitions[partition].append(features[offset : offset + seconds])
            label_seqs[partition].append((seconds, anomalous))
            offset += seconds

    # Concatenate per partition.
    n_feat = len(canonical_cols)

    def _stack(parts):
        return np.concatenate(parts, axis=0) if parts else np.empty((0, n_feat))

    train = _stack(partitions["train"])
    calib = _stack(partitions["calib"])
    test = _stack(partitions["test"])
    valid = _stack(partitions["valid"])

    if train.size == 0:
        raise ValueError("Training partition is empty after split. Cannot normalize.")

    # Zero out columns that are constant in the training (baseline) data across
    # every partition. Without this, the eps-regularised divisor in normalize()
    # amplifies any calib/test variation in such columns by 1/eps (~10000x),
    # drowning out signal from the well-behaved features.
    keep = train.max(axis=0) != train.min(axis=0)
    if not keep.all():
        constant_cols = [c for c, k in zip(canonical_cols, keep) if not k]
        print(f"  zeroing {len(constant_cols)} baseline-constant columns: {constant_cols}")
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

    def _build_labels(arr, name):
        out = np.zeros_like(arr)
        row = 0
        for seconds, anomalous in label_seqs.get(name, []):
            if anomalous:
                out[row : row + seconds] = 1
            row += seconds
        return out

    labels = _build_labels(test, "test")
    valid_labels = _build_labels(valid, "valid")
    print(
        "   Anomaly lables: {int(labels.sum() / n_feat_lbl) if labels.size else 0}/{labels.shape[0]} test rows, "
        f"{int(valid_labels.sum() / n_feat_lbl) if valid_labels.size else 0}/{valid_labels.shape[0]} valid rows"
    )
    print(f"  Feature columns ({len(canonical_cols)}): {canonical_cols}")

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
    all_paths = [p for paths, _ in csv_groups for p in paths]
    print(f"Processed {len(all_paths)} CSV(s) as TOL -> {folder}/")
    print(
        f"  train.npy: {train.shape}, test.npy: {test.shape}, calib.npy: {calib.shape}, valid.npy: {valid.shape}, labels.npy: {labels.shape}"
    )
