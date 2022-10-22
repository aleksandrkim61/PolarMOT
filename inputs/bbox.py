from __future__ import annotations
import math
from typing import Dict, Optional, List, Tuple, Any, Mapping
from collections import namedtuple
from abc import abstractmethod, ABC
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils.data_classes import Box

from transform.transformation import to_homogeneous, inverse_rigid_transform
import transform.nuscenes as nu_transform
import utils.utils_geometry as utils_geometry
from tracking.utils_tracks import correct_new_angle_and_diff
from dataset_classes.nuscenes.classes import id_from_name


Bbox2d = namedtuple('Bbox2d', 'x1 y1 x2 y2')


class ProjectsToCam(ABC):
    @abstractmethod
    def bbox_2d_in_cam(self, cam: str) -> Optional[Bbox2d]:
        pass


class Bbox3d(ProjectsToCam):
    """ corners_3d format. Facing forward: (0-1-4-5) = forward
      4 -------- 5
     /|         /|
    7 -------- 6 .
    | |        | |
    . 0 -------- 1
    |/         |/
    3 -------- 2
    """

    def __init__(self, bbox_coordinates, gt_track_id=None,
                 confidence=None, obs_angle=None, seg_class_id=None, velocity=None, info=None,
                 bbox_2d_in_cam: Dict[str, Optional[Bbox2d]] = None,
                 corners_3d_yaw: Tuple[np.ndarray, float] = None,
                 instance_id=None):
        self.original_coordinates = bbox_coordinates  # 7 elements: (x y z rotation-around-y l(x) w(z) h(y))
        self.gt_track_id = int(gt_track_id) if gt_track_id is not None else -1
        self.confidence = confidence
        self.obs_angle = obs_angle
        self.velocity = velocity
        self.seg_class_id = seg_class_id
        self.info = info
        self.instance_id = instance_id
        self._bbox_2d_in_cam: Dict[str, Optional[Bbox2d]
                                   ] = bbox_2d_in_cam if bbox_2d_in_cam is not None else {}

        if corners_3d_yaw is None:
            self.original_yaw = bbox_coordinates[3]
            self.corners_3d = utils_geometry.compute_box_3d(bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2],
                                                            bbox_coordinates[4], bbox_coordinates[5], bbox_coordinates[6],
                                                            yaw=self.original_yaw)
        else:
            (self.corners_3d, self.original_yaw) = corners_3d_yaw

        self.kf_coordinates = self.original_coordinates.copy()
        self.kf_coordinates[3] = self.original_yaw

    @classmethod
    def from_pointrcnn(cls, coordinates, gt_track_id=None, info=None, det_to_track_seg_class=None, instance_id=None) -> "Bbox3d":
        if info is None:
            return cls(coordinates, gt_track_id=gt_track_id)
        else:
            return cls(coordinates, confidence=info[6], obs_angle=info[0],
                       seg_class_id=det_to_track_seg_class[info[1]][1],
                       bbox_2d_in_cam={"image_02": info[2:6]},
                       gt_track_id=gt_track_id, info=info, instance_id=instance_id)

    @classmethod
    def from_pointgnn(cls, bbox_coordinates, confidence, seg_class_id,
                      bbox_2d_in_cam: Dict[str, Optional[Bbox2d]], info=None, gt_track_id=None, instance_id=None) -> "Bbox3d":
        return cls(bbox_coordinates, confidence=confidence, obs_angle=info[0],
                   seg_class_id=seg_class_id, bbox_2d_in_cam=bbox_2d_in_cam, info=info, gt_track_id=gt_track_id, instance_id=instance_id)

    @classmethod
    def from_nu_det(cls, det: Mapping, gt_track_id=None, convert_to_kitti: bool = True, instance_id=None) -> "Bbox3d":
        center = det["translation"]
        size = det["size"]
        orientation = Quaternion(det["rotation"])
        score = det["detection_score"]
        name = det["detection_name"]
        velocity = det["velocity"]

        bbox_nu = Box(center, size, orientation, score=score, velocity=velocity, name=name)
        if convert_to_kitti:
            return Bbox3d.from_nu_box_convert(bbox_nu, gt_track_id, instance_id)
        else:
            return Bbox3d.from_nu_box_no_conversion(bbox_nu, gt_track_id, instance_id)

    @classmethod
    def from_nu_box_convert(cls, bbox_nu: Box, gt_track_id=None, instance_id=None) -> "Bbox3d":
        coordinates_expected = nu_transform.convert_nu_bbox_coordinates_to_kitti(
            bbox_nu.center, bbox_nu.wlh, bbox_nu.orientation)
        name_parts = bbox_nu.name.split(".")
        detection_name = name_parts[1] if len(name_parts) > 1 else name_parts[0]
        return cls(coordinates_expected, gt_track_id,
                   confidence=bbox_nu.score, velocity=bbox_nu.velocity,
                   seg_class_id=id_from_name(detection_name), instance_id=instance_id)

    @classmethod
    def from_nu_box_no_conversion(cls, bbox_nu: Box, gt_track_id=None, instance_id=None) -> "Bbox3d":
        angle_around_vertical = quaternion_yaw(bbox_nu.orientation)
        coordinates = np.array([*bbox_nu.center, angle_around_vertical, *bbox_nu.wlh])
        # Bbox3d wants l, w, h => x, z, y
        detection_name = bbox_nu.name.split(".")[1]
        corners_3d_yaw = (bbox_nu.corners().T, angle_around_vertical)
        return cls(coordinates, gt_track_id, confidence=1.0,
                   seg_class_id=id_from_name(detection_name),
                   corners_3d_yaw=corners_3d_yaw, instance_id=instance_id)

    def bbox_2d_in_cam(self, cam: str) -> Optional[Bbox2d]:
        bbox = self._bbox_2d_in_cam.get(cam, None)
        return Bbox2d(*bbox) if bbox is not None else None

    def clear_2d(self):
        self._bbox_2d_in_cam.clear()

    def get_indices_of_points_inside(self, points, margin=0.0):
        """ Find indices of points inside the bbox

        :param points: 3D points in rectified camera coordinates
        :param margin: margin for the bbox to include boundary points, defaults to 0.0
        :return: indices of input points that are inside the bbox
        """
        # Axis align points and bbox boundary for easier filtering
        # This is 4x faster than `points = np.dot(points, self.rotation_matrix)`
        points = (self.rotation_matrix.T @ points.T).T

        rotated_first_corner = self.rotation_matrix.T @ self.corners_3d[6]
        rotated_last_corner = self.rotation_matrix.T @ self.corners_3d[0]

        mask_coordinates_inside = np.logical_and(
            points >= rotated_first_corner - margin, points <= rotated_last_corner + margin)
        return np.flatnonzero(np.all(mask_coordinates_inside, axis=1))

    def transform(self, transformation, angle_around_y):
        assert transformation is not None, 'Requested None transformation'
        corners_and_center = np.vstack((self.corners_3d, self.kf_coordinates[:3].reshape(1, -1)))  # Nx3
        transformed_corners_and_center = to_homogeneous(
            corners_and_center)  # Nx4, but usually it should be 4xN
        transformed_corners_and_center = transformation @ transformed_corners_and_center.T
        transformed_corners_and_center = transformed_corners_and_center.T

        self.corners_3d = transformed_corners_and_center[:-1, :-1]

        self.kf_coordinates[:3] = transformed_corners_and_center[-1, :-1]
        self.kf_coordinates[3] += angle_around_y

    def inverse_transform(self, transformation, angle_around_y):
        assert transformation is not None, 'Requested None reverse transformation'
        self.transform(inverse_rigid_transform(transformation), -angle_around_y)

    def reset_kf_coordinates(self):
        self.kf_coordinates = self.original_coordinates.copy()

    @property
    def rotation_matrix(self):
        return utils_geometry.get_rotation_matrix_around_y(self.kf_coordinates[3])

    @property
    def centroid_original(self):
        return np.array([self.original_coordinates[0], self.original_coordinates[1], self.original_coordinates[2]])

    @property
    def distance_to_ego(self):
        return np.linalg.norm(self.kf_coordinates[(0, 2), ])  # only x-z plane - ignore elevation

    def same_track(self, other: Bbox3d) -> bool:
        return self.gt_track_id is not None and self.gt_track_id == other.gt_track_id

    @property
    def gnn_features(self) -> np.ndarray:
        "returns x,z and theta coordinates in the tracking coordinate frame, i.e. after transformation to ego/world"
        return self.kf_coordinates[(0, 2, 3), ]

    @property
    def xz_coordinates(self) -> np.ndarray:
        "returns x,z coordinates in the tracking coordinate frame, i.e. after transformation to ego/world"
        return self.kf_coordinates[(0, 2), ]

    @property
    def rotation_y(self) -> np.ndarray:
        "returns rotation around y-axis in the tracking coordinate frame, i.e. after transformation to ego/world"
        return self.kf_coordinates[3]

    def difference(self, other: Bbox3d, include_dims: bool) -> np.ndarray:
        if include_dims:
            coord_diff = self.kf_coordinates - other.kf_coordinates
        else:  # only x,y,z,theta
            coord_diff = self.kf_coordinates[:4] - other.kf_coordinates[:4]
        # need to normalize angle difference, think about optimizing, maybe just normalize the difference
        coord_diff[3] = correct_new_angle_and_diff(self.kf_coordinates[3], other.kf_coordinates[3])[1]
        # ignore y coordinate - elevation does not matter, but can keep height difference
        return np.delete(coord_diff, 1)

    def __sub__(self, other: Bbox3d) -> np.ndarray:
        raise NotImplementedError("Use bbox.difference()")

    def __isub__(self, other: Bbox3d):
        raise NotImplementedError(
            "Do not implement this function to avoid bugs, subtraction returns a vector")

    def __str__(self):
        # center_str = f"xzy = [{self.original_coordinates[0]:>6.2f}, {self.original_coordinates[2]:>6.2f}, {self.original_coordinates[1]:>6.2f}] "
        # lwh_str    = f"lwh = [{self.original_coordinates[4]:>5.2f}, {self.original_coordinates[5]:>5.2f}, {self.original_coordinates[6]:>5.2f}] "
        # return f"id={self.gt_track_id:<4}: {center_str:<28} {lwh_str} angle = {self.original_yaw:<4.3f}"
        center_str = f"xzy = [{self.kf_coordinates[0]:>6.2f}, {self.kf_coordinates[2]:>6.2f}, {self.kf_coordinates[1]:>6.2f}] "
        lwh_str = f"lwh = [{self.kf_coordinates[4]:>5.2f}, {self.kf_coordinates[5]:>5.2f}, {self.kf_coordinates[6]:>5.2f}] "
        return f"track={self.gt_track_id:<4} instance_id={str(self.instance_id):<4}: {center_str:<28} {lwh_str} angle = {self.kf_coordinates[3]:<4.3f}"
        # return str(self.kf_coordinates)
    
    def __repr__(self):
        return str(self)

    def save(self, path_to_frame_folder):
        with open(Path(path_to_frame_folder) / str(self.instance_id), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path_to_file):
        with open(Path(path_to_file), 'rb') as handle:
            return pickle.load(handle)
