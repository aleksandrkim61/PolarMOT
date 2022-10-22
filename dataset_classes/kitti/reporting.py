from typing import Optional, IO, Mapping, Any, Iterable, List
import inputs.detections_2d as detections_2d
from inputs.bbox import Bbox3d


def write_to_mot_file(frame_name: str, bboxes: Iterable[Bbox3d],
                      mot_3d_file: IO,
                      mot_2d_from_3d_only_file: Optional[IO]) -> None:
    mot_3d_results_str, mot_2d_results_str = "", ""
    tracking_3d_format = "%d %d %s 0 0 %f -1 -1 -1 -1 %f %f %f %f %f %f %f %f\n"
    tracking_2d_format = "%d %d %s 0 0 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n"

    for bbox in bboxes:
        if bbox.gt_track_id is None or bbox.gt_track_id == -1:
            continue

        assert bbox.seg_class_id is not None
        track_type = detections_2d.SEG_TO_TRACK_CLASS[bbox.seg_class_id]

        bbox3d_coords = bbox.original_coordinates
        if bbox3d_coords is not None:
            res_3d = (tracking_3d_format % (int(frame_name), bbox.gt_track_id, track_type, bbox.obs_angle,
                                            bbox3d_coords[0], bbox3d_coords[1], bbox3d_coords[2],
                                            bbox3d_coords[3], bbox3d_coords[4], bbox3d_coords[5], bbox3d_coords[6], bbox.confidence))
            mot_3d_results_str += res_3d

        if mot_2d_from_3d_only_file is not None:
            bbox2d = bbox.bbox_2d_in_cam("image_02")
            if bbox2d is not None:
                res_2d = (tracking_2d_format % (int(frame_name), bbox.gt_track_id, track_type,
                                                bbox2d[0], bbox2d[1], bbox2d[2], bbox2d[3], bbox.confidence))
                mot_2d_results_str += res_2d

    mot_3d_file.write(mot_3d_results_str)
    if mot_2d_from_3d_only_file is not None:
        mot_2d_from_3d_only_file.write(mot_2d_results_str)
