import os
import numpy as np
import pandas as pd
import AaltoAD.utils as utils


def test_cut_array_center_slice():
    arr = np.arange(20).reshape(10, 2)
    # 50% should return middle portion -> 2*window rows (window=round(10*0.5*0.5)=2)
    cut = utils.cut_array(0.5, arr)
    assert cut.shape == (4, 2)
    # check that the returned array is centered
    assert np.array_equal(cut, arr[3:7])


def test_getresults2_basic():
    df = pd.DataFrame({
        'FN': [1, 2],
        'FP': [3, 4],
        'TP': [5, 6],
        'TN': [7, 8],
        'precision': [0.5, 0.8],
        'recall': [0.6, 0.9],
    })
    res = utils.getresults2(df, result=None)
    # sums
    assert res['FN'] == 3
    assert res['FP'] == 7
    # precision/recall should be means
    assert abs(res['precision'] - ((0.5 + 0.8) / 2)) < 1e-8
    assert abs(res['recall'] - ((0.6 + 0.9) / 2)) < 1e-8
    # f1* should be computed
    expected_f1 = 2 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    assert abs(res['f1*'] - expected_f1) < 1e-8
