import numpy as np
import AaltoAD.merlin as merlin_mod
import AaltoAD.constants as const


def test_dist_and_getsub():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert merlin_mod.dist(a, b) == 0.0
    arr = np.arange(10)
    sub = merlin_mod.getsub(arr, 3, 2)
    assert np.array_equal(sub, np.array([2, 3, 4]))


def test_check_returns_labels_and_matrix(monkeypatch):
    # make merlin module's constants deterministic for this test
    monkeypatch.setattr(const, 'cvp', 3)
    monkeypatch.setattr(const, 'percentile_merlin', 50)

    # simple 2-column time series with one column noisy
    t = np.array([[0.0, 0.0],
                  [0.0, 10.0],
                  [0.0, 0.0],
                  [0.0, 10.0],
                  [0.0, 0.0]])
    dummy_pred = np.zeros(t.shape[0], dtype=int)
    pred_vec, labels_mat = merlin_mod.check(t, dummy_pred)
    assert pred_vec.shape[0] == t.shape[0]
    assert labels_mat.shape == t.shape
    # labels_mat should be 0/1
    assert set(np.unique(labels_mat)).issubset({0, 1})


def test_get_result_simple():
    # simple pred and label arrays
    pred = np.array([0, 1, 0, 1])
    labels = np.array([0, 1, 0, 0])
    res = merlin_mod.get_result(pred, labels)
    assert isinstance(res, dict)
    for k in ('f1', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN', 'ROC/AUC'):
        assert k in res


def test_merlin_smoke_run():
    # Smoke test for merlin(): create a time series with a clear discord subsequence
    t = np.zeros((150, 1))
    t[50:60] = 100.0
    # call merlin with small window sizes to keep runtime small
    dbest, DFinal = merlin_mod.merlin(t, 3, 6)
    # ensure returned structures are the expected types
    assert isinstance(dbest, tuple)
    assert isinstance(DFinal, list)
    # each element in DFinal should be a triple (start, L, distance)
    if DFinal:
        assert len(DFinal[0]) == 3
