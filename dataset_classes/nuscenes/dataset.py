from __future__ import annotations
import os
from typing import Optional, List, Dict, Set, Any, Iterable, Sequence, Mapping

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from dataset_classes.mot_dataset import MOTDataset
from dataset_classes.nuscenes.sequence import MOTSequenceNuScenes
from configs.local_variables import NUSCENES_DATA_DIR
import dataset_classes.nuscenes.reporting as reporting
from inputs.bbox import Bbox3d


class MOTDatasetNuScenes(MOTDataset):
    ALL_SPLITS = {"train", "val", "test", "train_detect", "train_track",
                  "mini_train", "mini_val"}

    def __init__(self, work_dir: str, det_source: str, seg_source: str, version: str, params: Mapping):

        super().__init__(work_dir, det_source, seg_source, params)
        """ Initialize dataset-level NuScenes object

        :param version: version of NuScenes to use e.g. "v1.0-mini", "v1.0-trainval", "v1.0-test"
        """
        print(f"Parsing NuScenes {version} ...")
        self.nusc = NuScenes(version=version, dataroot=NUSCENES_DATA_DIR, verbose=True)
        self.splits: Set[str] = set(s for s in self.ALL_SPLITS if any(
            part_s in version for part_s in s.split("_")))
        self.sequences_by_name: Dict[str, Any] = {
            scene["name"]: scene for scene in self.nusc.scene
        }
        self.splits_to_scene_names: Dict[str, List[str]] = create_splits_scenes(nusc=self.nusc)
        print("Done parsing")

        self.version = version
        self.reset()

    def reset(self, only_submission: bool = False) -> None:
        self.submission: Dict[str, Dict[str, Any]] = {"meta": {"use_camera": False,
                                                               "use_lidar": True,
                                                               "use_radar": False,
                                                               "use_map": False,
                                                               "use_external": False},
                                                      "results": {}}
        if not only_submission:
            self.detections_3d: Dict[str, List[Bbox3d]] = {}

    def sequence_names(self, split: str) -> List[str]:  # overrides base method
        self.assert_split_exists(split)
        return self.splits_to_scene_names[split]

    def get_sequence(self, split: str, sequence_name: str) -> MOTSequenceNuScenes:  # overrides base method
        self.assert_sequence_in_split_exists(split, sequence_name)
        split_dir = os.path.join(self.work_dir, split)
        return MOTSequenceNuScenes(self.det_source, self.seg_source, split_dir, split,
                                   self.nusc, self.sequences_by_name[sequence_name],
                                   self.submission, self.detections_3d,
                                   self.params, self.work_dir)

    def save_all_mot_results(self, folder_name: str) -> None:  # overrides base method
        reporting.save_to_json_file(self.submission, folder_name, self.version)

    def scene_names_for_area(self, area_substring: str):
        log_tokens = [log["token"] for log in self.nusc.log if area_substring.lower() in log["location"].lower()]
        return [scene["name"] for scene in self.nusc.scene if scene["log_token"] in log_tokens]
