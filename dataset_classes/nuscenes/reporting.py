import os
import ujson as json
import time
from typing import IO, Any, Dict, Iterable

import numpy as np
from pyquaternion import Quaternion

from inputs.bbox import Bbox3d
from dataset_classes.nuscenes.classes import name_from_id
from transform.nuscenes import convert_kitti_bbox_coordinates_to_nu


def build_results_dict(bbox: Bbox3d, frame_token: str) -> Dict[str, Any]:
    assert bbox.gt_track_id is not None
    bbox3d_coords = bbox.original_coordinates  # should be reordered to [h, w, l, x, y, z, theta]
    assert bbox3d_coords is not None
    center, wlh, rotation = convert_kitti_bbox_coordinates_to_nu(bbox3d_coords)

    track_dict: Dict[str, Any] = {"sample_token": frame_token}
    track_dict["translation"] = center.tolist()
    track_dict["size"] = wlh.tolist()
    track_dict["rotation"] = rotation.elements.tolist()
    velocity = bbox.velocity
    track_dict["velocity"] = list(velocity) if velocity is not None else [1.0, 1.0]
    track_dict["tracking_id"] = str(bbox.gt_track_id)
    track_dict["tracking_name"] = name_from_id(bbox.seg_class_id)
    track_dict["tracking_score"] = bbox.confidence
    track_dict["yaw"] = bbox3d_coords[6]
    return track_dict


def add_results_to_submit(submission: Dict[str, Dict[str, Any]], frame_token: str,
                      bboxes: Iterable[Bbox3d]) -> None:
    assert frame_token not in submission["results"], submission["results"][frame_token]
    submission["results"][frame_token] = []

    for bbox in bboxes:
        if bbox.gt_track_id is None or bbox.gt_track_id == -1:
            continue
        submission["results"][frame_token].append(build_results_dict(bbox, frame_token))

    if len(submission["results"][frame_token]) == 0:
        print(f"Nothing tracked for {frame_token}")


def save_to_json_file(submission: Dict[str, Dict[str, Any]],
                             folder_name: str, version: str) -> None:
    print(f"Frames tracked: {len(submission['results'].keys())}")
    results_file = os.path.join(folder_name, (version + "_tracking.json"))
    with open(results_file, 'w') as f:
        json.dump(submission, f, indent=4)
