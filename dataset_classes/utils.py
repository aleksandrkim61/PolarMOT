from typing import List

from inputs.bbox import Bbox3d


MAX_DETS_PER_FRAME = 10000


def assign_ids_to_frame_detections(frame_i: int, detections: List[Bbox3d]) -> None:
    """Assign unique ids to parsed Bboxes.
    frame number = det.instance_id // MAX_DETS_PER_FRAME
    detection in frame = det.instance_id % MAX_DETS_PER_FRAME

    :param detections: a list of bboxes
    """
    for det_i, det in enumerate(detections):
        if det is not None:
            det.instance_id = MAX_DETS_PER_FRAME * frame_i + det_i
