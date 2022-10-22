import numpy as np
from numba import njit


@njit
def first_true_index_or_minus_one(array):
    """ For 1D arrays, this should be the fastest implementation 
    than even np functions which do not stop early 
    """
    for i, value in enumerate(array):
        if value:
            return i
    return -1


@njit
def last_true_index_or_minus_one(array):
    """ For 1D arrays, this should be the fastest implementation 
    than even np functions which do not stop early 
    """
    for index_from_end, value in enumerate(array[::-1]):
        if value:
            return len(array) - 1 - index_from_end
    return -1
