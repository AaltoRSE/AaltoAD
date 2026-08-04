"""Tests for normalize() and dataset preprocessors.

Only datasets whose source files are tracked in the repository are exercised.
SWaT_*, MSDS, WADI, FTSD are excluded because their source data is not committed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from AaltoAD.preprocessing.utils import normalize


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(REPO_ROOT / "data")


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

def test_normalize_basic_range():
    a = np.array([0.0, 5.0, 10.0])
    out, min_a, max_a = normalize(a)
    assert min_a == 0.0
    assert max_a == 10.0
    # First element exactly 0, last element just under 1 because of eps.
    assert out[0] == 0.0
    assert 0.99 < out[-1] < 1.0
    assert np.all((out >= 0) & (out <= 1))


def test_normalize_returns_min_max_when_none():
    a = np.array([2.0, 4.0, 6.0])
    out, min_a, max_a = normalize(a)
    assert min_a == 2.0
    assert max_a == 6.0


def test_normalize_uses_supplied_min_max():
    train = np.array([0.0, 1.0, 2.0])
    test = np.array([1.0, 3.0])  # outside train range
    _, min_a, max_a = normalize(train)
    out_test, _, _ = normalize(test, min_a, max_a)
    # test value 3.0 is above train max 2.0 -> normalized > 1
    assert out_test[1] > 1.0
    # test value 1.0 sits in the middle of train range
    assert 0.49 < out_test[0] < 0.51


def test_normalize_constant_array_no_nan():
    """Constant column must not produce NaN/inf (the whole point of eps)."""
    a = np.array([3.0, 3.0, 3.0, 3.0])
    out, _, _ = normalize(a)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    # All zeros because numerator is 0 everywhere.
    assert np.allclose(out, 0.0)


def test_normalize_constanttrain_step_with_varying_test():
    """Non-zero test values for a constant-train column blow up by ~1/eps."""
    train = np.array([1.0, 1.0, 1.0])
    test = np.array([1.0, 2.0])
    _, min_a, max_a = normalize(train)
    out_test, _, _ = normalize(test, min_a, max_a)
    assert not np.isnan(out_test).any()
    # test value 1.0 maps to 0.0; test value 2.0 explodes.
    assert out_test[0] == 0.0
    assert out_test[1] > 1000  # ~1/eps


def test_normalize_2d_per_column_default():
    """2D input is normalized per-column by default: min/max are arrays of shape (n_features,)."""
    a = np.array([[0.0, 10.0], [5.0, 20.0]])
    out, min_a, max_a = normalize(a)
    assert np.array_equal(min_a, np.array([0.0, 10.0]))
    assert np.array_equal(max_a, np.array([5.0, 20.0]))
    assert out.shape == a.shape
    # Each column independently in [0, 1).
    assert np.all(out.min(axis=0) == 0.0)
    assert np.all(out.max(axis=0) < 1.0)


def test_normalize_2d_shared_flag():
    """shared=True falls back to whole-array min/max (scalar)."""
    a = np.array([[0.0, 10.0], [5.0, 20.0]])
    out, min_a, max_a = normalize(a, shared=True)
    assert min_a == 0.0
    assert max_a == 20.0
    assert out.shape == a.shape


def test_normalizetrain_step_lands_in_unit_range():
    """After fitting on itself, every train element must be in [0, 1)."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-50, 50, size=200)
    out, _, _ = normalize(a)
    assert out.min() == 0.0
    assert 0.99 < out.max() < 1.0
    assert np.all((out >= 0) & (out < 1))


def test_normalize_negative_range():
    a = np.array([-10.0, -5.0, 0.0])
    out, min_a, max_a = normalize(a)
    assert min_a == -10.0
    assert max_a == 0.0
    assert out[0] == 0.0
    assert 0.99 < out[-1] < 1.0


def test_normalize_signature_preserves_input_shape():
    a = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
    out, _, _ = normalize(a)
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Dataset preprocessor smoke tests.
# Each runs the loader against the repo's tracked source data and asserts that
# the produced .npy files are well-formed: no NaN/inf, train approximately in
# [0, 1], and labels/train/test shapes are consistent.
# ---------------------------------------------------------------------------

def _assert_cleantrain_step(path):
    a = np.load(path)
    assert a.size > 0
    assert not np.isnan(a).any(), f"NaN in {path}"
    assert not np.isinf(a).any(), f"inf in {path}"
    # Train must land in [0, 1] (allow tiny epsilon slack).
    assert a.min() >= -1e-6, f"train min {a.min()} below 0 in {path}"
    assert a.max() <= 1.0 + 1e-6, f"train max {a.max()} above 1 in {path}"


def _assert_clean_finite(path):
    a = np.load(path)
    assert a.size > 0
    assert not np.isnan(a).any(), f"NaN in {path}"
    assert not np.isinf(a).any(), f"inf in {path}"


def test_load_NAB(tmp_path):
    from AaltoAD.preprocessing.nab import load_NAB
    out = tmp_path / "NAB"
    out.mkdir()
    load_NAB(str(out), data_folder=DATA_DIR)
    npy_files = list(out.glob("*.npy"))
    assert npy_files, "load_NAB produced no .npy files"
    for f in npy_files:
        if f.name.endswith("train.npy"):
            _assert_cleantrain_step(f)
        else:
            _assert_clean_finite(f)


def test_load_synthetic(tmp_path):
    from AaltoAD.preprocessing.synthetic import load_synthetic
    out = tmp_path / "synthetic"
    out.mkdir()
    np.random.seed(0)  # the loader uses np.random
    load_synthetic(str(out), data_folder=DATA_DIR)
    train = np.load(out / "train.npy")
    test = np.load(out / "test.npy")
    labels = np.load(out / "labels.npy")
    _assert_cleantrain_step(out / "train.npy")
    _assert_clean_finite(out / "test.npy")
    assert labels.shape == test.shape
    assert set(np.unique(labels.astype(int))) <= {0, 1}


def test_load_UCR(tmp_path):
    from AaltoAD.preprocessing.ucr import load_UCR
    out = tmp_path / "UCR"
    out.mkdir()
    load_UCR(str(out), data_folder=DATA_DIR)
    train_files = list(out.glob("*train.npy"))
    assert train_files, "load_UCR produced no train.npy files"
    for f in out.glob("*train.npy"):
        _assert_cleantrain_step(f)
    for f in out.glob("*_test.npy"):
        _assert_clean_finite(f)
    for f in out.glob("*_labels.npy"):
        _assert_clean_finite(f)


def test_load_MBA(tmp_path):
    pytest.importorskip("openpyxl")
    from AaltoAD.preprocessing.mba import load_MBA
    out = tmp_path / "MBA"
    out.mkdir()
    load_MBA(str(out), data_folder=DATA_DIR)
    _assert_cleantrain_step(out / "train.npy")
    _assert_clean_finite(out / "test.npy")
    labels = np.load(out / "labels.npy")
    test = np.load(out / "test.npy")
    assert labels.shape[0] == test.shape[0]


def test_load_SMAP(tmp_path):
    from AaltoAD.preprocessing.smap_msl import load_SMAP_MSL
    out = tmp_path / "SMAP"
    out.mkdir()
    load_SMAP_MSL(str(out), "SMAP", data_folder=DATA_DIR)
    train_files = list(out.glob("*_train.npy"))
    assert train_files, "load_SMAP_MSL(SMAP) produced no _train.npy files"
    for f in train_files:
        _assert_cleantrain_step(f)
    for f in out.glob("*_test.npy"):
        _assert_clean_finite(f)


def test_load_MSL(tmp_path):
    from AaltoAD.preprocessing.smap_msl import load_SMAP_MSL
    out = tmp_path / "MSL"
    out.mkdir()
    load_SMAP_MSL(str(out), "MSL", data_folder=DATA_DIR)
    train_files = list(out.glob("*_train.npy"))
    assert train_files, "load_SMAP_MSL(MSL) produced no _train.npy files"
    for f in train_files:
        _assert_cleantrain_step(f)


def test_load_SMD(tmp_path, monkeypatch):
    """SMD's loader writes to a relative 'processed/SMD/...' path via
    load_and_save's default-arg DEFAULT_OUTPUT_FOLDER. Run from tmp_path with
    a symlink to the repo's data folder so output lands in the temp dir.
    """
    from AaltoAD.preprocessing.smd import load_SMD

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").symlink_to(REPO_ROOT / "data")
    (tmp_path / "processed" / "SMD").mkdir(parents=True)

    load_SMD(str(tmp_path / "processed" / "SMD"))
    smd_dir = tmp_path / "processed" / "SMD"
    train_files = list(smd_dir.glob("*train.npy"))
    assert train_files, "load_SMD produced no train.npy files"
    for f in train_files:
        a = np.load(f)
        assert not np.isnan(a).any()
        assert not np.isinf(a).any()
