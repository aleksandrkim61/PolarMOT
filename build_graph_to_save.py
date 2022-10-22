import time
from itertools import product
from typing import List, Iterable, Mapping
import os
from pathlib import Path
import datetime
import random
import gc
from copy import deepcopy
import argparse

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from pytorch_lightning.utilities.seed import seed_everything
import ujson as json

from dataset_classes.kitti.mot_kitti import MOTDatasetKITTI
from dataset_classes.nuscenes.dataset import MOTDatasetNuScenes
from dataset_classes.mot_sequence import MOTSequence
from configs.params import TRAIN_SEQ, VAL_SEQ, TRACK_TRAIN_SEQ, TRACK_VAL_SEQ, build_params_dict, KITTI_BEST_PARAMS, NUSCENES_BEST_PARAMS, variant_name_from_params
from configs.local_variables import KITTI_WORK_DIR, NUSCENES_WORK_DIR, MOUNT_PATH
import inputs.utils as input_utils
from dataset_classes.kitti.classes import KITTIClasses
from dataset_classes.nuscenes.classes import NuScenesClasses
import data.graph_construction as graph_construction
from configs.data_graph_build_params import max_edge_distances, xz_stds, theta_stds, bbox_add_p, num_bboxes_to_always_add
from utils.io import folder_name_from_params


def build_data(mot_dataset_, clip_length, data_params_, split: str, full: bool):
    if isinstance(mot_dataset, MOTDatasetKITTI):
        dataset_split = "training"
        if split == "train":
            seq_names = TRACK_TRAIN_SEQ if full else ["0002"]
        elif split == "val":
            seq_names = TRACK_VAL_SEQ if full else ["0006"]
        else:
            raise NotImplementedError(f"Unknown split: {split}")
#     elif isinstance(mot_dataset, MOTDatasetNuScenes):
    else:
        # "train", "val", "train_track" (subset of train), "test", "mini_train", "mini_val"
        dataset_split = split if full or "mini" in split else f"mini_{split}"
        seq_names = None

    return graph_construction.from_dataset(mot_dataset_, dataset_split, seq_names, **data_params_)


parser = argparse.ArgumentParser()
parser.add_argument("class_str", type=str, help="Class to run for, e.g. 'car', 'pedestrian', etc.")
parser.add_argument("-max_edge_length", type=int, default=-1,
                    help="Maximum frame gap in connected nodes, e.g. -1, 5")
parser.add_argument('--online', default=False, action='store_true')
parser.add_argument('--offline', dest="online", action='store_false')
parser.add_argument('--not_link_past', default=False, action='store_true')
parser.add_argument('--cartesian', default=False, action='store_true')
parser.add_argument('--no_delta', default=False, action='store_true')
parser.add_argument('--mini', default=False, action='store_true')
parser.add_argument('--no_val', default=False, action='store_true')
parser.add_argument('--no_train', default=False, action='store_true')
parser.add_argument("-area_filter", type=str, default="",
                    help="Area from which to take data, everything else will be ignored")
args = parser.parse_args()
seg_class_id = NuScenesClasses[args.class_str]
print("Launching for", seg_class_id)


CLIP_LENGTH = 20 if args.online else 11  # 20 for online, 11 for offline
seed = 124

data_full = True
num_augmentations = 1  # If online, it will be +1, need more!

annotated = True
max_edge_length = args.max_edge_length  # -1 to connect all, 5 otherwise
include_dims = False

deltas_only = not args.no_delta
use_polar_attr = not args.cartesian

online_only = args.online
link_past_tracks = not args.not_link_past and online_only
link_past_tracks_mode = "last"

data_params = {
    "seed": seed,
    "clip_len": CLIP_LENGTH,
    "seg_class_id": seg_class_id,
    "annotated": annotated,
    "max_edge_length": args.max_edge_length,
    "area_filter": args.area_filter,

    "include_dims": include_dims,
    "deltas_only": deltas_only,
    "max_edge_distance": max_edge_distances[seg_class_id],
    "online_only": online_only,
    "use_polar_attr": use_polar_attr,
    "link_past_tracks": link_past_tracks,
    "link_past_tracks_mode": link_past_tracks_mode,
    "connect_intraframe": True,

    "bbox_drop_p": 0,
    "frame_drop_p": 0,
    "xz_std": xz_stds[seg_class_id],
    "theta_std": theta_stds[seg_class_id],
    "lwh_std": 0,

    "bbox_add_p": bbox_add_p[seg_class_id],
    "num_bboxes_to_always_add": num_bboxes_to_always_add[seg_class_id],
}

if not deltas_only:
    print("Requesting relative features not adjusted for time! Distace thresholds will not work as well")

save_folder_name = f"{datetime.datetime.now().strftime('%y-%m-%d')}_aug1"
save_folder_name += folder_name_from_params(args.max_edge_length, deltas_only, use_polar_attr, False)
save_folder_name += f"_{'on' if online_only else 'off'}line"
if online_only:
    save_folder_name += f"_{('link' + link_past_tracks_mode) if link_past_tracks else 'nolink'}"
save_folder_name += f"_{CLIP_LENGTH}_nodrop"
if args.area_filter:
    save_folder_name += f"_{args.area_filter}"
save_folder_name += f"_{seg_class_id.name}"
save_dir = Path("/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval") / save_folder_name
save_dir.mkdir(parents=True, exist_ok=True)
data_params_to_save = deepcopy(data_params)
data_params_to_save["seg_class_id"] = data_params_to_save["seg_class_id"].value
with open(save_dir / "data_params.json", 'w') as f:
    json.dump(data_params_to_save, f, indent=4)

# mot_dataset = MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
#                               det_source=input_utils.POINTGNN_T3,
#                               seg_source=input_utils.TRACKING_BEST,
#                               params=KITTI_BEST_PARAMS)
mot_dataset = MOTDatasetNuScenes(work_dir=NUSCENES_WORK_DIR,
                                 det_source=input_utils.CENTER_POINT,
                                 seg_source=input_utils.MMDETECTION_CASCADE_NUIMAGES,
                                 params=NUSCENES_BEST_PARAMS,
                                 version="v1.0-mini" if args.mini and args.no_val else "v1.0-trainval")  # "v1.0-trainval", "v1.0-mini"
mot_dataset.params["do_not_init_model"] = True

save_dir_processed = save_dir / "processed"
save_dir_processed.mkdir(parents=True, exist_ok=True)

# Train set
if not args.no_train:
    data_train_list = []
    seed_everything(data_params["seed"])

    for i in range(num_augmentations):
        data_train_list.extend(build_data(mot_dataset, CLIP_LENGTH, data_params,
                                          f"{'mini_' if args.mini else ''}train", data_full))
        gc.collect()

    data, slices = InMemoryDataset.collate(data_train_list)
    del data_train_list
    gc.collect()

    torch.save((data, slices), save_dir_processed / f"{'mini_' if args.mini else ''}train.pt")
    del data
    del slices
    gc.collect()

# Val set
if not args.no_val:
    data_val_list = []
    seed_everything(data_params["seed"])

    for i in range(num_augmentations):
        data_val_list.extend(build_data(mot_dataset, CLIP_LENGTH, data_params, "val", data_full))
        gc.collect()

    del mot_dataset
    gc.collect()

    data, slices = InMemoryDataset.collate(data_val_list)
    del data_val_list
    gc.collect()

    torch.save((data, slices), save_dir_processed / f"val.pt")
