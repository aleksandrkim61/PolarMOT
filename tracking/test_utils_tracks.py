import math

import numpy as np

import tracking.utils_tracks as utils_tracks


# I know, I know...not pretty
angles_diffs_to_test = [(30, 320, -70), (320, 30, +70), (-40, 30, +70), (30, -40, -70)]
angles_diffs_to_test += [(-40, 140, 180), (-40, 130, 170), (-40, 150, -170), (-40, -20, 20)]
angles_diffs_to_test += [(140, -40, -180), (130, -40, -170), (150, -40, 170), (-20, -40, -20)]
angles_diffs_to_test += [(120, 200, 80), (120, 300, 180), (120, 290, 170), (120, 310, -170)]
angles_diffs_to_test += [(170, -170, 20), (-170, 170, -20), (170, 200, 30), (200, 170, -30)]
angles_diffs_to_test += [(80, 100, 20), (100, 80, -20), (250, -80, 30), (-110, -80, 30), (-80, -110, -30)]
angles_diffs_to_test += [(440, 460, 20), (460, 80, -20), (250, -80, 30), (-470, -80, 30), (-80, -110, -30)]


def test_angle_arrays_diff():
    start_angles_array = np.deg2rad([t[0] for t in angles_diffs_to_test])
    target_angles_array = np.deg2rad([t[1] for t in angles_diffs_to_test])
    expected_result = np.deg2rad([t[2] for t in angles_diffs_to_test])

    pi_expected = np.isclose(np.absolute(expected_result), np.pi)
    output = utils_tracks.compute_angle_diff_arrays(start_angles_array, target_angles_array)
    np.testing.assert_almost_equal(np.absolute(output[pi_expected]), np.pi,
                                   err_msg=np.absolute(output[pi_expected]))
    np.testing.assert_almost_equal(output[~pi_expected], expected_result[~pi_expected])
    np.testing.assert_array_less(np.absolute(output), np.full_like(output, np.pi + 1e-7))


def test_angle_diff():
    for start_angle, target_angle, expected_diff in angles_diffs_to_test:
        diff = utils_tracks.compute_angle_diff(np.deg2rad(start_angle), np.deg2rad(target_angle))
        diff = np.rad2deg(diff)
        error_message = f"{start_angle} to {target_angle} should be {expected_diff} not {diff}"

        if abs(expected_diff) != 180:
            np.testing.assert_almost_equal(diff, expected_diff, err_msg=error_message)
        else:
            np.testing.assert_almost_equal(abs(diff), abs(expected_diff), err_msg=error_message)
    assert abs(diff) <= 180 + 1e-7, f"{start_angle} to {target_angle} = {diff} is not in [-PI, PI] range"


# Fails
# def test_angle_diff_old():
#     angles_diffs_old_to_test = [(30, 320, -70), (320, 30, +70), (-40, 30, +70), (30, -40, -70)]
#     angles_diffs_old_to_test += [(-40, 140, 0), (-40, 130, -10), (-40, 150, -10), (-40, -20, 20)]
#     angles_diffs_old_to_test += [(140, -40, 0), (130, -40, -10), (150, -40, 10), (-20, -40, -20)]
#     for start_angle, target_angle, expected_diff in angles_diffs_old_to_test:
#         _, diff = utils_tracks.correct_new_angle_and_diff(np.deg2rad(start_angle), np.deg2rad(target_angle))
#         diff = np.rad2deg(diff)
#         error_message = f"{start_angle} to {target_angle} should be {expected_diff} not {diff}"

#         if abs(expected_diff) != 180:
#             np.testing.assert_almost_equal(diff, expected_diff, err_msg=error_message)
#         else:
#             np.testing.assert_almost_equal(abs(diff), abs(expected_diff), err_msg=error_message)
#     assert abs(diff) <= 180 + 1e-7, f"{start_angle} to {target_angle} = {diff} is not in [-PI, PI] range"


def test_normalize_angle():
    angles_to_normalize = [5, 359, 360, 361, -180, -90, -400, 500, 1100]
    expected_normalized = [5, 359, 0,  1, 180, 270, 320, 140, 20]
    assert len(angles_to_normalize) == len(expected_normalized)
    angles_to_normalize = np.deg2rad(angles_to_normalize)
    expected_normalized = np.deg2rad(expected_normalized)
    outs = np.fromiter((utils_tracks.normalize_angle(angle) for angle in angles_to_normalize), dtype=float)
    np.testing.assert_allclose(outs, expected_normalized,
                               err_msg=f"{angles_to_normalize} should be {expected_normalized}, not {np.rad2deg(outs)}")


def test_normalize_array_of_angles():
    angles_to_normalize = [5, 359, 360, 361, -180, -90, -400, 500, 1100]
    expected_normalized = [5, 359, 0,  1, 180, 270, 320, 140, 20]
    assert len(angles_to_normalize) == len(expected_normalized)
    angles_to_normalize = np.deg2rad(angles_to_normalize)
    expected_normalized = np.deg2rad(expected_normalized)
    normalized_array = utils_tracks.normalize_array_of_angles(angles_to_normalize)
    np.testing.assert_allclose(normalized_array, expected_normalized,
                               err_msg=f"{angles_to_normalize} should be {expected_normalized}, not {np.rad2deg(normalized_array)}")


def test_compute_rotation_around_y():
    sqrt3 = np.sqrt(3.0)
    xs = np.array([[2, -2, -2, 2], [3, 0, -3, 0],
                   [sqrt3, sqrt3, -sqrt3, -sqrt3], [1, 1, -1, -1]])
    zs = np.array([[2, 2, -2, -2], [0, -3, 0, 3],
                   [1, -1, -1, 1], [sqrt3, -sqrt3, -sqrt3, sqrt3]])
    expected_rotations = np.array([[-45, -135, 135, 45], [-0, 90, -180, -90],
                                  [-30,  30, 150, -150], [-60, 60, 120, -120]])
    assert xs.shape == zs.shape == expected_rotations.shape
    np.testing.assert_allclose(utils_tracks.compute_rotation_around_y(xs, zs), np.deg2rad(expected_rotations))
