from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Iterable, Mapping, Dict, Any, Optional, IO, Sequence
import datetime
from pathlib import Path

import numpy as np

import inputs.bbox as bbox
import tracking.tracking_manager as tracking_manager
import utils.io as io
import dataset_classes.mot_frame as mot_frame
from inputs.detection_2d import Detection2D
from objects.fused_instance import FusedInstance
from transform.transformation import Transformation
from configs.params import variant_name_from_params
import dataset_classes.utils as utils
from dataset_classes.common import DatasetClassEnum


class MOTSequence(ABC):
    def __init__(self, det_source: str, seg_source: str, split_dir: str, name: str,
                 frame_names: Sequence[str], params: Mapping, work_dir: str):
        self.det_source = det_source
        self.seg_source = seg_source
        self.split_dir = split_dir
        self.name = name
        self.frame_names = frame_names
        self.params = params
        self.work_dir = work_dir

        # Image size for each camera - needed for 3D->2D projections. The dict is set in dataset-specific classes
        self.img_shape_per_cam: Dict[str, Any] = {}

        # Detections 3D {frame_name: [bboxes_3d]}
        self._dets_3d_per_frame: Dict[str, List[bbox.Bbox3d]] = {}

        # Detections 2D {frame_name: {cam_name: [bboxes_3d]}}
        self.dets_2d_multicam_per_frame: Dict[str, Dict[str, List[Detection2D]]] = {}

        # need to set its Transformation object and img_shape_per_cam in subclasses
        self.mot = tracking_manager.TrackManager(self.class_enums_to_track, self.params, self.work_dir)

        det_seg_source_folder_name = f'{self.det_source}_{self.seg_source}'
        self.work_split_input_dir = os.path.join(self.split_dir, det_seg_source_folder_name)
        self.tracking_res_dir = os.path.join(self.work_split_input_dir, 'mot')
        self.bbox_instances_dir = os.path.join(self.work_split_input_dir, "bbox_instances", self.name)

    ##########################################################
    # Evaluation

    def perform_tracking_for_eval(self) -> Dict[str, Any]:
        mot_3d_file = io.create_writable_file_if_new(
            self.get_results_folder_name(self.params["run_name"], "3d"), self.name)
        if len(self.classes_to_track) == 2:  # only report 2D for KITTI
            mot_2d_from_3d_file = io.create_writable_file_if_new(
                self.get_results_folder_name(self.params["run_name"], "2d_projected_3d"), self.name)
        else:
            mot_2d_from_3d_file = None

        run_info: Dict[str, Any] = defaultdict(float)

        if mot_3d_file is None:
            print(f'Sequence {self.name} already has results. Skipped')
            print('=====================================================================================')
            return run_info

        # run_info["mot_3d_file"] = str(mot_3d_file.parent)
        # run_info["mot_2d_from_3d_file"] = str(mot_2d_from_3d_file.parent) if mot_2d_from_3d_file else ""
        run_info["mot_3d_file"] = mot_3d_file.name.split(self.name)[0]
        run_info["mot_2d_from_3d_file"] = mot_2d_from_3d_file.name.split(
            self.name)[0] if mot_2d_from_3d_file else ""

        for frame_i, frame_name in enumerate(self.frame_names):
            if frame_i % 100 == 0:
                print(f'Processing frame {frame_name}')

            frame = self.get_frame(frame_name)
            frame.load_detections_3d_if_needed()  # Load before timing tracking
            bboxes_to_report = frame.perform_tracking(run_info)

            start_reporting = time.time()
            self.report_mot_results(frame.name, bboxes_to_report, mot_3d_file, mot_2d_from_3d_file)
            run_info["total_time_reporting"] += time.time() - start_reporting

        self.save_mot_results(mot_3d_file, mot_2d_from_3d_file)
        self.save_ego_motion_transforms_if_new()
        return run_info

    def report_offline_tracking(self, target_class,
                                instance_id_to_final_track_id: Mapping[int, int],
                                instance_id_to_track_score: Mapping[int, float],
                                suffix: str = "", annotated=False) -> Dict[str, Any]:
        mot_3d_file = io.create_writable_file_if_new(
            self.get_results_folder_name(f"{suffix}", "3d"), self.name)

        run_info: Dict[str, Any] = defaultdict(int)

        if mot_3d_file is None:
            print(f'Sequence {self.name} already has results. Skipped')
            print('=====================================================================================')
            return run_info

        run_info["mot_3d_file"] = mot_3d_file.name.split(self.name)[0]

        reorder_back = [6, 5, 4, 0, 1, 2, 3]  # from [x,y,z,theta,l,w,h] to [h, w, l, x, y, z, theta]
        # TODO: add a threshold for the multiplied score as well, e.g. det_score=0.2, track_score=0.2 to take care of 0.4*0.5 - barely associated
        # Actually no, low scores are also useful to judge track viability
        for frame_i, frame_name in enumerate(self.frame_names):
            if frame_i % 100 == 0:
                print(f'Processing frame {frame_name}')

            frame = self.get_frame(frame_name)
            if not instance_id_to_final_track_id:
                self.report_mot_results(frame.name, [], mot_3d_file)
                continue

            bboxes = frame.bboxes_3d if not annotated else frame.bbox_3d_annotations(world=True)

            for bbox in bboxes:
                track_id = instance_id_to_final_track_id.get(bbox.instance_id, None)
                # TODO: report dets without a track with a low score_mult to trade IDs for Recall
                if bbox.seg_class_id != target_class or track_id is None:
                    continue
                bbox.gt_track_id = track_id
                bbox.original_coordinates = bbox.original_coordinates[reorder_back]
                bbox.confidence *= instance_id_to_track_score.get(bbox.instance_id, 1)

            start_reporting = time.time()
            self.report_mot_results(frame.name, bboxes, mot_3d_file)
            run_info["total_time_reporting"] += time.time() - start_reporting

        self.save_mot_results(mot_3d_file)
        self.save_ego_motion_transforms_if_new()
        return run_info

    def get_results_folder_name(self, folder_prefix: str, suffix: str):
        folder_suffix_full = f"{variant_name_from_params(self.params)}_{folder_prefix}_{suffix}"
        return f"{self.tracking_res_dir}_{folder_suffix_full}"

    ##########################################################
    # Lazy getters for frame-specific data

    def get_segmentations_for_frame(self, frame_name: str) -> Dict[str, List[Detection2D]]:
        """ Return a dict of Detection2D for each camera for the requested frame"""
        if not self.dets_2d_multicam_per_frame:
            self.dets_2d_multicam_per_frame = self.load_detections_2d()
        return self.dets_2d_multicam_per_frame.get(frame_name, defaultdict(list))

    def get_bboxes_for_frame(self, frame_name: str) -> List[bbox.Bbox3d]:
        """ Return a list of bbox.Bbox3d for the requested frame"""
        return self.dets_3d_per_frame.get(frame_name, [])

    @property
    def dets_3d_per_frame(self) -> Dict[str, List[bbox.Bbox3d]]:
        """ Return a list of bbox.Bbox3d for the requested frame"""
        if not self._dets_3d_per_frame:
            self._dets_3d_per_frame = self.load_detections_3d()
            self.assign_ids_to_detections(self._dets_3d_per_frame)
        return self._dets_3d_per_frame

    def assign_ids_to_detections(self, detections_dict: Dict[str, List[bbox.Bbox3d]]) -> None:
        """Assign unique ids to parsed Bboxes.
        frame number = det.instance_id // utils.MAX_DETS_PER_FRAME
        detection in frame = det.instance_id % utils.MAX_DETS_PER_FRAME

        :param detections_dict: a list of bboxes for each frame
        """
        if len(detections_dict) <= len(self.frame_names):  # only dets for this sequence - KITTI
            print("Assigning instance_id to KITTI dets")
            for frame_i, frame_name in enumerate(self.frame_names):
                utils.assign_ids_to_frame_detections(frame_i, detections_dict.get(frame_name, []))
        else:  # dets for all sequences - NuScenes, sort by frame key just to be sure
            print("Assigning instance_id to NuScenes dets")
            for frame_i, (frame_name, dets) in enumerate(sorted(detections_dict.items(), key=lambda kv: kv[0])):
                utils.assign_ids_to_frame_detections(frame_i, dets)

    ##########################################################
    # Required methods and fields that need to be overridden by subclasses
    # This sadly results in some extra code, but is the best way to ensure compile-time errors

    @abstractmethod
    def load_ego_motion_transforms(self) -> None: pass

    @abstractmethod
    def save_ego_motion_transforms_if_new(self) -> None: pass

    @abstractmethod
    def load_detections_3d(self) -> Dict[str, List[bbox.Bbox3d]]: pass

    @abstractmethod
    def load_detections_2d(self) -> Dict[str, Dict[str, List[Detection2D]]]: pass

    @abstractmethod
    def get_frame(self, frame_name: str) -> mot_frame.MOTFrame: pass

    @property
    @abstractmethod
    def transformation(self) -> Transformation: pass

    @property
    @abstractmethod
    def cameras(self) -> List[str]: pass

    @property
    @abstractmethod
    def camera_default(self) -> str: pass

    @property
    @abstractmethod
    def classes_to_track(self) -> List[int]: pass

    @property
    @abstractmethod
    def class_enums_to_track(self) -> List[DatasetClassEnum]: pass

    @abstractmethod
    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[FusedInstance],
                           mot_3d_file: IO,
                           mot_2d_from_3d_only_file: Optional[IO] = None) -> None:
        pass

    @abstractmethod
    def save_mot_results(self, mot_3d_file: IO,
                         mot_2d_from_3d_file: Optional[IO] = None) -> None:
        pass
