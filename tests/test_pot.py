import numpy as np
import TranAD.pot as pot


def test_adjust_predicts_basic():
    score = np.array([0.1, 0.8, 0.2, 0.9])
    label = np.array([0, 1, 0, 1])
    pred = pot.adjust_predicts(score, label, threshold=0.5)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == label.shape
    # where score > 0.5 should be predicted True
    assert pred[1] == True
    assert pred[3] == True
    assert pred[0] == False


def test_pot_eval_with_dummy_spot(monkeypatch):
    # small synthetic arrays
    init_score = np.array([0.1, 0.2, 0.05])
    score = np.array([0.2, 0.6, 0.1])
    label = np.array([0, 1, 0])

    class DummySPOT:
        def __init__(self, q):
            self.q = q

        def fit(self, init_score_, score_):
            self.init = init_score_
            self.score = score_

        def initialize(self, level=None, min_extrema=False, verbose=False):
            self.level = level

        def run(self, dynamic=False):
            # return a thresholds array so pot_eval computes a threshold
            return {"thresholds": np.array([0.6]), "alarms": []}

    # monkeypatch SPOT used inside pot module
    monkeypatch.setattr(pot, "SPOT", DummySPOT)

    res, preds = pot.pot_eval(init_score, score, label)
    assert isinstance(res, dict)
    # expected keys
    for k in ("f1", "precision", "recall", "TP", "TN", "FP", "FN", "ROC/AUC", "threshold"):
        assert k in res
    assert isinstance(preds, np.ndarray)
    assert preds.shape == score.shape
