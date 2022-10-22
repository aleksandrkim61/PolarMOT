import time
from typing import Iterable, List, Dict, Set, Optional, Sequence, Any, Sequence, Mapping, Deque, Tuple
from pathlib import Path
from collections import deque

import numpy as np
from pytorch_lightning import Trainer
import torch
from torch_geometric.data import Data

from tracking.data_association import (
    associate_instances_to_tracks_3d_iou,
    associate_instances_to_tracks_2d_iou,
    match_multicam, CamDetectionIndices
)
from tracking.tracks import Track
from collections import defaultdict
import dataset_classes.nuscenes.classes as nu_classes
from utils.utils_geometry import project_bbox_3d_to_2d
from inputs.bbox import Bbox3d, Bbox2d
from models.graph_tracker_offline import GraphTrackerOffline
import data.graph_construction as graph_construction
from dataset_classes.utils import MAX_DETS_PER_FRAME
from utils.io import get_best_ckpt
from dataset_classes.common import DatasetClassEnum


# from [x,y,z,theta,l,w,h] to [h, w, l, x, y, z, theta]
REORDER_COORDINATES_TO_ORIGINAL = [6, 5, 4, 0, 1, 2, 3]


class TrackManager(object):
    def __init__(self, class_enums_to_track: Iterable[DatasetClassEnum], params: Mapping, work_dir: str):
        self.class_enums_to_track = class_enums_to_track
        self.classes_to_track = [c.value for c in class_enums_to_track]
        self.work_dir = work_dir
        self.params = params
        if not self.params.get("do_not_init_model", False):
            self._parse_params()
            self.graph_trackers: Dict[DatasetClassEnum, Any] = {}
            self.init_graph_tracker()  # to immediately make sure CUDA works
            # maxlen for deque will automatically discard objects that are too old
            self.clips_per_class: Dict[DatasetClassEnum, Deque[List[Track]]] = {
                class_enum: deque(maxlen=params["max_frame_age"] + 1)
                for class_enum in self.class_enums_to_track}
        self.track_id_latest = 1  # evaluations expect positive track ids

        self.instance_to_track_id_and_score: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
        self.instance_ids_to_report: Set[int] = set()
        # Map each track to its last delta distance and number of frames since last link

        self.track_id_to_hits: Dict[int, int] = defaultdict(int)

    def _parse_params(self):
        self.min_hits = self.params["min_hits"]
        self.max_past_dets = self.params["max_past_dets"]
        self.track_pred_thresholds = self.params["track_pred_thresholds"]
        self.track_initial_mult = self.params["track_initial_mult"]
        self.max_edge_distances = self.params["max_edge_distances"]
        self.max_edge_distance_multiplier = self.params["max_edge_distance_multiplier"]
        # set a minimal confidence threshold to report, otherwise don't report at all - to discard noisy FP detections
        # Low confidense dets can be confirmed by a high track pred score, but if not confirmed, then its a FP
        self.det_scores_to_report = self.params["det_scores_to_report"]

    def init_graph_tracker(self):
        for class_enum in self.class_enums_to_track:
            ckpt_path = get_best_ckpt(Path(self.work_dir) / self.params["pretrained_runs_folder"],
                                      class_enum.name)
            model_to_test = GraphTrackerOffline.load_from_checkpoint(ckpt_path,
                                                                     hparams_file=str(ckpt_path.parent.parent / "hparams.yaml"))
            model_to_test.cuda().eval()
            self.graph_trackers[class_enum] = model_to_test

    @torch.inference_mode()
    def predict_latest_tracks(self, graph_data: Data, class_enum: DatasetClassEnum, run_info: Dict):
        min_score_to_store = self.track_initial_mult[class_enum]
        _start_time = time.time()

        edge_index = graph_data.edge_index.T.numpy()
        edge_attr = graph_data.edge_attr.numpy()
        node_to_instance_ids = graph_data.instance_ids.numpy()

        assert self.graph_trackers
        cont_predictions: np.ndarray = self.graph_trackers[class_enum].predict_step(graph_data.cuda())[0].squeeze().cpu().numpy()
        cont_predictions = np.atleast_1d(cont_predictions)
        assert len(cont_predictions) == len(edge_index), f"{cont_predictions}\n{graph_data}"
        run_info["total_time_track_predicting"] += time.time() - _start_time

        _start_time = time.time()
        assigned_tracks: Set[int] = set()
        for i in reversed(cont_predictions.argsort()):
            score: float = cont_predictions[i]
            if score <= min_score_to_store:
                break

            source_instance_id, target_instance_id = node_to_instance_ids[edge_index[i]]
            # This is no longer guaranteed because NuScenes frames are out of order
            # assert target_instance_id > source_instance_id, f"Connected {source_instance_id} to {target_instance_id}"
            # print(score, source_instance_id, target_instance_id)

            if target_instance_id in self.instance_ids_to_report:  # have not assigned this det yet
                # Should not have matched this det with any earlier ones yet
                assert target_instance_id not in self.instance_to_track_id_and_score

                track_id_to_assign, _ = self.instance_to_track_id_and_score.get(source_instance_id,
                                                                                (None, None))
                if track_id_to_assign in assigned_tracks:  # already matched this existing track
                    continue

                # this track was not assigned to in this frame yet
                #   Looks like a model trained on linked past automatically learned this constraint,
                #   can write about it - a relaxed problem formulation was used to learn a constrained solution
                #   only thorough the graph construction
                
                # if score >= self.track_pred_thresholds[class_enum]:  # confirmed association to an earlier detection
                #     if track_id_to_assign is not None: 
                #         assigned_tracks.add(track_id_to_assign)  # remember that this track_id was taken
                #     # else:  # matched to an earlier detection that was not assigned a track / reported
                #     self.instance_to_track_id_and_score[target_instance_id] = (track_id_to_assign, score)
                # else:
                #     # No association, so just remember the highest link score to multiply with det_score to get final reported confidence
                #     self.instance_to_track_id_and_score[target_instance_id] = (None, score)
                
                if score < self.track_pred_thresholds[class_enum]:  # do not take that track_id - score is too low
                    track_id_to_assign = None
                elif track_id_to_assign is not None:  # confirmed association - score is high enough
                    assigned_tracks.add(track_id_to_assign)  # remember that this track_id was taken
                self.instance_to_track_id_and_score[target_instance_id] = (track_id_to_assign, score)

                # Only match each of the current detections once
                self.instance_ids_to_report.remove(target_instance_id)
                if not len(self.instance_ids_to_report):  # already assigned all latest dets
                    break
            # Do not put any more logic here, there is an early `continue` exit above
        run_info["total_time_track_assigning"] += time.time() - _start_time

    def update(self, latest_bboxes: Iterable[Bbox3d], run_info: Dict) -> List[Bbox3d]:
        """ Matches latest frame's detections with existing tracks and manages their lifecycle.
        Should be called for each frame even with empty detections

        :param latest_bboxes: list of Bbox3d objects from the latest frame
        """
        if not self.graph_trackers:
            self.init_graph_tracker()

        bboxes_to_report: List[Bbox3d] = []
        for class_enum, clip in self.clips_per_class.items():
            _start_time = time.time()
            bboxes_current = [bbox for bbox in latest_bboxes if bbox.seg_class_id == class_enum.value]
            clip.append(bboxes_current)  # bboxes from older frames will be pushed out automatically
            if not bboxes_current:
                continue  # no dets in the latest frame
            # A set of latest instant ids - only these can be assigned new track ids
            # Has to be done before predicting with GNN
            self.instance_ids_to_report = {bbox.instance_id for bbox in bboxes_current}
            run_info["total_time_list_building"] += time.time() - _start_time

            # construct graph object
            _start_time = time.time()
            # TODO: Optimize graph construction - the bottleneck
            graph_data = graph_construction.from_clip(clip, run_info, annotated=False,
                                                      max_edge_distance=self.max_edge_distance_multiplier*self.max_edge_distances[class_enum],
                                                      **self.params)
            if graph_data is None:
                continue  # no viable edges from the latest clip
            run_info["total_time_graph_construction"] += time.time() - _start_time

            _start_time = time.time()
            # Link latest bboxes to existing ones
            self.predict_latest_tracks(graph_data, class_enum, run_info)
            run_info["total_time_predict_latest_tracks"] += time.time() - _start_time

            _start_time = time.time()
            # update latest bboxes with their matched track ids or assign new ones
            for bbox in bboxes_current:
                target_instance_id = bbox.instance_id
                track_id_to_assign, track_score = self.instance_to_track_id_and_score.get(target_instance_id,
                                                                                          (None, None))
                # TODO: do not assign track_id if the score is below 0.5, but multiply the score
                # seems like the model gives a pretty good link score to show if the node is good or not
                # Accepting track links with 0.3/0.2 seems bad, but somehow works. Might be because the mult is too high

                # needs to start a new track if its confidence is high enough
                if track_id_to_assign is None and bbox.confidence >= self.det_scores_to_report[class_enum.value - 1]:
                    run_info["new_tracks_count"] += 1
                    # precision is similar to offline case, but recall and IDs are very bad.
                    # Must not be reporting enough detections with high enough scores?
                    # Or reporting bad detections with high scores forcing best MOTA to be at lower recall
                    # detections should have already been filtered when parsing

                    # assign initial_mult if no link score available
                    track_score = track_score or self.track_initial_mult[class_enum]
                    
                    track_id_to_assign = self.track_id_latest
                    self.track_id_latest += 1
                    self.instance_to_track_id_and_score[target_instance_id] = (track_id_to_assign, None)

                if track_id_to_assign is not None:
                    if self.track_id_to_hits[track_id_to_assign] >= self.min_hits[class_enum.value - 1]:
                        bbox.gt_track_id = track_id_to_assign
                        # Final confidence = det score (frame-level info) * matching score (sequence-level info) 
                        bbox.confidence *= track_score

                    self.track_id_to_hits[track_id_to_assign] += 1
            run_info["total_time_track_confidence_assigning"] += time.time() - _start_time

            # Report tracks for the current class
            _start_time = time.time()
            for bbox in bboxes_current:
                bbox.original_coordinates = bbox.original_coordinates[REORDER_COORDINATES_TO_ORIGINAL]
            run_info["total_time_coordinate_reorder"] += time.time() - _start_time
            bboxes_to_report.extend(bboxes_current)

            self.cull_older_dets(class_enum, clip)
        return bboxes_to_report

    def cull_older_dets(self, class_enum, clip):
        track_lens: Dict[int, int] = defaultdict(int)
        for det_list in reversed(clip):
            indices_to_delete = set()
            for i, det in enumerate(det_list):
                track_id = det.gt_track_id
                if track_id == -1:
                    continue

                if track_lens[track_id] >= self.max_past_dets[class_enum.value - 1]:
                    indices_to_delete.add(i)
                    continue

                track_lens[track_id] += 1

            det_list[:] = [det for i, det in enumerate(det_list) if i not in indices_to_delete]
