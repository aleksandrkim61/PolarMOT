from typing import Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from numba import njit

PI = np.pi
TWO_PI = 2 * np.pi


def normalize_angle(angle: float) -> float:
    """ Bring the angle to be in range [0, 2PI] """
    # angle = np.copysign(abs(angle) % TWO_PI, angle)  # now abs(angle) < TWO_PI
    angle = np.fmod(angle, TWO_PI)  # now abs(angle) < TWO_PI
    if angle > 0:
        return angle
    angle += TWO_PI  # to handle negatives
    return angle % TWO_PI  # to correct positives that got an extra TWO_PI

def normalize_array_of_angles(angles_array: np.ndarray) -> np.ndarray:
    # now abs(angles_array) < TWO_PI
    angles_array = np.fmod(angles_array, TWO_PI)
    negative_mask = angles_array < 0
    angles_array[negative_mask] += TWO_PI  # to handle negatives
    # to correct positives that got an extra TWO_PI
    angles_array[negative_mask] = angles_array[negative_mask] % TWO_PI
    return angles_array


def normalize_array_of_angles_inplace(angles_array: np.ndarray) -> None:
    """ Same as above but inplace. Separate to avoid condition for numba """
    np.fmod(angles_array, TWO_PI, angles_array)
    negative_mask = angles_array < 0
    angles_array[negative_mask] += TWO_PI  # to handle negatives
    angles_array[negative_mask] = angles_array[negative_mask] % TWO_PI


def compute_angle_diff(start_angle: float, target_angle: float) -> float:
    """ Returns the difference between two angles, such that start_angle + difference = target_angle.
    The output is in the range [-PI, PI]
    Do not use this directly, use the array-based function below instead
    """
    original_diff = normalize_angle(target_angle) - normalize_angle(start_angle)
    if abs(original_diff) > PI:
        diff = TWO_PI - abs(original_diff)
        if original_diff > 0:
            diff *= -1
    else:
        diff = original_diff
    return diff


def compute_angle_diff_arrays(start_angles: np.ndarray, target_angles: np.ndarray) -> np.ndarray:
    """ Returns the difference between two arrays of angles, such that start_angles + difference = target_angles.
    The output for each pair is in the range [-PI, PI]
    """
    diff = normalize_array_of_angles(target_angles) - normalize_array_of_angles(start_angles)
    fix_angles_from_difference_inplace(diff)
    return diff


def fix_angles_from_difference_inplace(diff: np.ndarray) -> None:
    """ Fixes, inpace, given angles. 
    Expects that they were produced by subtracting normalized angles from each other, so max(abs(diff)) <= 2PI
    See `compute_angle_diff_arrays(start_angles, target_angles)` above
    This function can be called after stacking/broadcasting two arrays of angles to get pairwise difference, for example.
    The output is in the range [-PI, PI]
    """
    abs_original_diff = np.absolute(diff)
    to_loop_mask = abs_original_diff > PI
    to_flip_if_loop_mask = diff > 0
    diff[to_loop_mask]
    diff[to_loop_mask] = TWO_PI - abs_original_diff[to_loop_mask]
    diff[to_loop_mask & to_flip_if_loop_mask] *= -1


def compute_rotation_around_y(x_coords: np.ndarray, z_coords: np.ndarray) -> np.ndarray:
    """ Computes clock-wise rotation around y from the X-axis (KITTI orientation convention)
    for each vector defined by a pair of corresponding cells row in the input arrays

    :param x_coords: an array (any shape) of x coordinates for input vectors
    :param z_coords: an array (same shape as x) of z coordinates for input vectors
    :return: rotation around Y-axis in the range [-PI, PI]
    """
    assert x_coords.shape == z_coords.shape, f"Shapes are not the same {x_coords.shape}, {z_coords.shape}"
    return -np.arctan2(z_coords, x_coords)


def correct_new_angle_and_diff(current_angle: float, new_angle_to_correct: float) -> Tuple[float, float]:
    """ Return an angle equivalent to the new_angle_to_correct with regards to difference to the current_angle
    Calculate the difference between two angles [-PI/2, PI/2]

    This function is ugly but works. 
    The other `compute_angle_diff` function returns results in [-PI, PI], i.e. it preserves flips, 
    while this one takes the smallest difference after trying to flip
    """
    abs_diff = normalize_angle(new_angle_to_correct) - normalize_angle(current_angle)
    # TODO: should normalize result before returning
    # fails for (-40, 360)
    # also the difference returned fails for (30, 320), should be -70, returns 70
     
    if abs(abs_diff) <= PI / 2:  # if in adjacent quadrants
        return new_angle_to_correct, abs_diff

    if abs(abs_diff) >= 3 * PI / 2:  # if in 1st and 4th quadrants and the angle needs to loop around
        abs_diff = TWO_PI - abs(abs_diff)
        if current_angle < new_angle_to_correct:
            return current_angle - abs_diff, abs_diff
        else:
            return current_angle + abs_diff, abs_diff

    # if the difference is > PI/2 and the new angle needs to be flipped
    return correct_new_angle_and_diff(current_angle, PI + new_angle_to_correct)


def default_kf_3d(is_angular: bool) -> KalmanFilter:
    if is_angular:  # add angular velocity to the state vector
        kf = KalmanFilter(dim_x=11, dim_z=7)
        kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
                         [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    else:
        kf = KalmanFilter(dim_x=10, dim_z=7)  # [x,y,z,theta,l,w,h] + [vx, vy, vz]
        kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                         [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

    # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
    kf.P[7:, 7:] *= 1000.
    kf.P *= 10.
    kf.Q[7:, 7:] *= 0.01
    kf.R *= 0.01  # measurement uncertainty (own addition)
    return kf
