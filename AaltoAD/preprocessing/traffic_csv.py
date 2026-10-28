"""Shared CSV parsing and segment-layout helpers for the traffic-style loaders (TOL, nettraffic).

Public API: parse_timestamp_column, find_timestamp_column, detect_column,
detect_ip_columns, shannon_entropy, validate_segments, baseline_mask.
"""

import re

import numpy as np
import pandas as pd


def parse_timestamp_column(series):
    """Parse a timestamp column and return integer unix-seconds (nullable Int64).

    Rules:
    - If values are numeric, detect scale (s, ms, us, ns) by magnitude and
      normalize to seconds, floored to integers.
    - If values are datetime strings, parse to datetimes and convert to integer seconds.
    - Returns a pandas Series of dtype `Int64` with unix seconds or NA.
    """
    non_na = series.dropna()
    if non_na.empty:
        raise ValueError("Timestamp column contains only nulls")
    first = non_na.iloc[0]
    try:
        is_numeric = isinstance(first, (int, float, np.integer, np.floating))
        if not is_numeric:
            fs = str(first).strip()
            is_numeric = bool(re.match(r"^\d+(\.\d+)?$", fs))
    except Exception:
        is_numeric = False
    if is_numeric:
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        if numeric.dropna().empty:
            # produce nullable Int64 of NAs
            numeric_seconds = pd.Series(
                [pd.NA] * len(numeric), index=numeric.index, dtype="Int64"
            )
        else:
            maxv = numeric.dropna().abs().max()
            if maxv > 1e17:
                div = 1e9
            elif maxv > 1e14:
                div = 1e6
            elif maxv > 1e11:
                div = 1e3
            else:
                div = 1.0
            secs = np.floor(numeric / div)
            numeric_seconds = (
                pd.Series(secs, index=numeric.index)
                .where(~pd.isna(secs))
                .astype("Int64")
            )
    else:
        dt = pd.to_datetime(series, errors="coerce")
        if dt.isna().all():
            raise ValueError("Unable to parse timestamps from column")
        numeric_seconds = pd.Series(
            (dt.view("int64") // 1_000_000_000), index=dt.index
        ).astype("Int64")
    print(
        f"TT Timestamp parsed: original='{first}' -> parsed={numeric_seconds.iloc[0]}"
    )
    return numeric_seconds


def find_timestamp_column(df, timestamp_candidates=None):
    """Find the timestamp column name in `df`.

    Returns the first column whose name matches a candidate (case-insensitive).
    Raises ValueError if none matches; the column is never guessed.
    """
    if timestamp_candidates is None:
        timestamp_candidates = ["timestamp", "time", "ts", "date", "datetime"]
    # exact name match (case-insensitive)
    for name in timestamp_candidates:
        for c in df.columns:
            if c.lower() == name:
                return c
    raise ValueError(
        f"No timestamp column found: expected one of {timestamp_candidates}, "
        f"got columns {list(df.columns)}. Rename the column in the CSV."
    )


def detect_column(df, candidates):
    """Return first column whose lower-cased name is in candidates, else None."""
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None


def detect_ip_columns(df):
    """Find src/dst IP columns in a traffic CSV, preferring IP columns over MAC columns."""
    ip_src_names = ("src_ip", "ip.src", "ip_src")
    ip_dst_names = ("dst_ip", "ip.dst", "ip_dst")
    src_col = next((c for c in df.columns if c.lower() in ip_src_names), None)
    dst_col = next((c for c in df.columns if c.lower() in ip_dst_names), None)
    if src_col is None or dst_col is None:
        for c in df.columns:
            cl = c.lower()
            if "eth" in cl or "mac" in cl:
                continue
            if src_col is None and "src" in cl:
                src_col = c
            if dst_col is None and "dst" in cl:
                dst_col = c
        if src_col is None or dst_col is None:
            for c in df.columns:
                if src_col is None and "ip" in c.lower():
                    src_col = c
                if dst_col is None and "ip" in c.lower():
                    dst_col = c
    return src_col, dst_col


def shannon_entropy(series):
    """Shannon entropy (bits) of a categorical Series."""
    p = series.value_counts(normalize=True)
    return -(p * np.log2(p + 1e-10)).sum()


def validate_segments(segments):
    """Check segments form a valid layout: each non-skip partition appears as one contiguous block.

    'skip' segments are unrestricted — they consume seconds from the CSV but produce
    no output, so multiple skip blocks in arbitrary positions are fine.
    """
    valid_partitions = {"train", "calib", "test", "valid", "skip"}
    seen_blocks = {}  # partition -> (first_idx, last_idx)
    for i, (partition, anomalous, seconds) in enumerate(segments):
        if partition not in valid_partitions:
            raise ValueError(
                f"Unknown partition {partition!r}; must be one of {sorted(valid_partitions)}"
            )
        if not isinstance(anomalous, bool):
            raise ValueError(
                f"Anomalous flag for segment {i} must be bool, got {type(anomalous).__name__}"
            )
        if anomalous and partition == "skip":
            raise ValueError(f"Segment {i}: 'skip' segments cannot be anomalous")
        if not isinstance(seconds, int) or seconds <= 0:
            raise ValueError(
                f"Seconds for segment {i} must be a positive int, got {seconds!r}"
            )
        if partition == "skip":
            continue  # skip is exempt from contiguity tracking
        if partition in seen_blocks:
            first, last = seen_blocks[partition]
            if i != last + 1:
                # Allow contiguity to be broken only by skip segments between blocks.
                for j in range(last + 1, i):
                    if segments[j][0] != "skip":
                        raise ValueError(
                            f"Partition {partition!r} is not contiguous: appears at index {first} and again at {i} "
                            f"after a gap. Group all segments of the same partition together "
                            f"(skip-only gaps allowed)."
                        )
                seen_blocks[partition] = (first, i)
            else:
                seen_blocks[partition] = (first, i)
        else:
            seen_blocks[partition] = (i, i)


def baseline_mask(df, segments, interval_col="ts_sec"):
    """Mask selecting rows whose ts_sec falls in any train segment of the segment list."""
    ts_min = df[interval_col].min()
    ranges = []
    offset = 0
    for partition, _, seconds in segments:
        if partition == "train":
            ranges.append((ts_min + offset, ts_min + offset + seconds))
        offset += seconds
    if not ranges:
        # No train segments — caller (multi-CSV) may still want IP stats from this
        # CSV if it's a pure-test source; return empty mask so it contributes nothing.
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for lo, hi in ranges:
        mask |= (df[interval_col] >= lo) & (df[interval_col] < hi)
    return mask
