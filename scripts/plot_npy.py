#!/usr/bin/env python3
"""Simple plotting for processed .npy time-series files.

Usage: python scripts/plot_tol.py path/to/processed_file.npy [--out-dir OUT] [--features 0,1,..|all] [--show]

Produces one PNG per feature (default: all features) in OUT/<dataset_name>/feature_{i}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot features from a processed .npy time-series file")
    p.add_argument("npy_file", help="Path to the .npy file (2D array T x D)")
    p.add_argument("--out-dir", default="plots", help="Base output directory for saved PNGs")
    p.add_argument("--features", default="all",
                   help="Comma-separated feature indices (e.g. 0,1,2) or 'all' (default)")
    p.add_argument("--show", action="store_true", help="Show plots interactively after saving")
    return p.parse_args()


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    return arr


def main() -> None:
    args = parse_args()
    p = Path(args.npy_file)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(2)

    arr = np.load(p)
    arr = ensure_2d(arr)
    T, D = arr.shape
    print(f"Loaded {p} with shape (T={T}, D={D})")

    # determine dataset name from parent folder if possible, else stem
    dataset_name = p.parent.name or p.stem
    out_base = Path(args.out_dir) / dataset_name
    out_base.mkdir(parents=True, exist_ok=True)

    # parse features
    if args.features.strip().lower() in ("all", "*"):
        feature_idxs = list(range(D))
    else:
        try:
            feature_idxs = [int(x) for x in args.features.split(",") if x.strip() != ""]
        except ValueError:
            print("Invalid --features list; use comma-separated integers or 'all'")
            sys.exit(2)

    # validate
    for i in feature_idxs:
        if i < 0 or i >= D:
            print(f"Feature index out of range: {i} (expected 0..{D-1})")
            sys.exit(2)

    x = np.arange(T)
    for i in feature_idxs:
        y = arr[:, i]
        plt.figure(figsize=(12, 3.5))
        plt.plot(x, y, lw=0.9)
        plt.xlabel("Sample index")
        plt.ylabel(f"Feature {i}")
        plt.title(f"{dataset_name} - feature {i} (shape {T}×{D})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_file = out_base / f"feature_{i}.png"
        plt.savefig(out_file, dpi=150)
        print(f"Saved {out_file}")
        if args.show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
