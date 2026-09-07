import numpy as np

from AaltoAD import constants
from AaltoAD.thresholds.point_adjust import calc_point2point, adjust_predicts, segment_latency
from AaltoAD.thresholds.pot_fit import fit_pot_threshold, pot_metrics

def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-5, expand_segments=False):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        expanded_segments (bool): Whether to expand the detected anomaly segments to include adjacent points.
    Returns:
        dict: pot result dict

    The tail used to fit the GPD is selected via constants.level: calibration
    points above that quantile are the excesses the GPD is fitted to.
    """
    if np.any(np.isnan(init_score)) or np.any(np.isnan(score)):
        pred = np.zeros_like(label)
        return pot_metrics(score, label, float('nan'), expand_segments), pred

    level = constants.level
    pot_th = fit_pot_threshold(init_score, score, q, level)
    if np.isnan(pot_th):
        pred = np.zeros_like(label)
        return pot_metrics(score, label, pot_th, expand_segments), pred

    # Threshold comes from the calibration data only: fit_pot_threshold fits
    # the GPD to the peaks of init_score and extrapolates the quantile at
    # risk q.
    raw_pred = score > pot_th
    if expand_segments:
        pred = adjust_predicts(score, label, pot_th)
    else:
        pred = raw_pred
    result = pot_metrics(score, label, pot_th, expand_segments)
    return result, np.array(pred)
