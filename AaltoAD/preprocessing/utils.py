"""
Functionality shared between multiple dataset preprocessing modules 
"""

import numpy as np
from AaltoAD.constants import DEFAULT_OUTPUT_FOLDER


def normalize(a, min_a=None, max_a=None, eps=1e-4, shared=False):
    """Normalize array to range [min_range, max_range] using symmetric scaling.

    Args:
            a (np.ndarray): Input array of any shape.
            min_a (float): Minimum value of input array for scaling. If None, uses a.min().
            max_a (float): Maximum value of input array for scaling. If None, uses a
            min_range (float): Minimum value of the target range.
            max_range (float): Maximum value of the target range.
            eps (float): Small value to avoid division by zero.

    Returns:
            np.ndarray: Normalized array with same shape, values in approximately [min_range, max_range].
    """

    axis = None if shared or np.ndim(a) < 2 else 0
    if min_a is None:
        min_a = np.nanmin(a, axis=axis)
    if max_a is None:
        max_a = np.nanmax(a, axis=axis)
    a = (a - min_a) / (max_a - min_a + eps)
    return a, min_a, max_a



