#!/usr/bin/env python
# coding: utf-8

import time
from typing import Dict
import os
from pathlib import Path
import gc
import argparse

import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from pytorch_lightning import Trainer

import dataset_classes.kitti.mot_kitti as mot_kitti
from dataset_classes.nuscenes.dataset import MOTDatasetNuScenes
from dataset_classes.kitti.classes import KITTIClasses
from dataset_classes.nuscenes.classes import NuScenesClasses
from configs.params import TRACK_VAL_SEQ, KITTI_BEST_PARAMS, NUSCENES_BEST_PARAMS, variant_name_from_params
from configs.local_variables import KITTI_WORK_DIR, NUSCENES_WORK_DIR
import inputs.utils as input_utils

from configs.model_paths import offline_models, no_sameframe_models
from models.graph_tracker_offline import GraphTrackerOffline
from models.node_models import ContextualNodeModel
import data.graph_construction as graph_construction
import evaluation.offline_utils as offline_utils
import evaluation.offline_processing as processing
from data.single_graph_dataset import SingleGraphDataset
from configs.data_graph_build_params import max_edge_distances
from configs.data_paths import det_graphs_val_det0, det_graphs_val_det02, det_graphs_val_det03
from utils.io import get_best_ckpt, folder_name_from_params
from configs.inference_params import track_pred_thresholds_offline, det_thresholds_offline


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="Dataset on which to run, i.e. nuscenes or kitti")
parser.add_argument("class_str", type=str, help="Class to run for, e.g. 'car', 'pedestrian', etc.")
parser.add_argument("-max_edge_length", type=int, default=-1,
                    help="Maximum frame gap in connected nodes, e.g. -1, 5")
parser.add_argument("-det_score_threshold", type=float, default=-1,
                    help="Custom detection score threshold to track, e.g. 0.2")
parser.add_argument("-reduced_score_threshold", type=float, default=-1,
                    help="Custom link score threshold after median/mean, e.g. 0.5")
parser.add_argument("-max_time_diff", type=int, default=99,
                    help="Maximum frame gap in linked detections, e.g. 11")
parser.add_argument("-mpn_steps", type=int, default=4,
                    help="Number of mpn steps in the model")
parser.add_argument("-max_edge_distance_multiplier", type=float, default=1,
                    help="Custom multiplier for default max connected edge distance")

parser.add_argument('--cartesian', default=False, action='store_true')
parser.add_argument('--no_delta', default=False, action='store_true')
parser.add_argument('--annotated', default=False, action='store_true')
parser.add_argument('--no_sameframe', default=False, action='store_true')
parser.add_argument('--mini', default=False, action='store_true')
parser.add_argument('--median', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument("-edge_mlps_count", type=int, default=3, help="Number of distinct node MLPs")
parser.add_argument("-node_aggr_sections", type=int, default=3, help="Number of distinct sections in Node aggregation")
parser.add_argument("-area_filter", type=str, default="",
                    help="Area where the tracking model was trained")
parser.add_argument("--synced_detector", default=False, action="store_true",
                    help="If the detector should be trained also on area_filter or mini like the tracker.")
args = parser.parse_args()

assert not (args.mini and args.area_filter), "both mini and area_filter given"
if args.synced_detector:
    assert args.mini or args.area_filter, "synced_detector but neither mini not area_filter given"

dataset_to_eval = args.dataset
assert dataset_to_eval in ("kitti", "nuscenes"), f"Invalid dataset chosen {dataset_to_eval}"
if dataset_to_eval == "nuscenes":
    seg_class_id = NuScenesClasses[args.class_str]
    batch_size = 2  # to avoid OOM with 4Gb
    split = "test" if args.test else "val"
elif dataset_to_eval == "kitti":
    seg_class_id = KITTIClasses[args.class_str]
    batch_size = 16
    split = "training"
print("Eval for", seg_class_id)

annotated = args.annotated
score_reduce_mode = "mean" if not args.median else "median"
det_score_threshold = args.det_score_threshold if args.det_score_threshold != -1 \
    else det_thresholds_offline[seg_class_id]
reduced_score_threshold = args.reduced_score_threshold if args.reduced_score_threshold != -1 \
    else track_pred_thresholds_offline[seg_class_id]  # 0.5/0.7
max_time_diff = args.max_time_diff  # 11

NUM_WORKERS = 4
save_root_dir = Path(f"/storage/slurm/kimal/graphmot_workspace/{dataset_to_eval}/{split}")

save_folder_name = f"dets_offline"
save_folder_name += folder_name_from_params(args.max_edge_length, not args.no_delta,
                                            not args.cartesian, args.no_sameframe, args.max_edge_distance_multiplier)
if args.synced_detector:
    save_folder_name += f"{'_mini' if args.mini else ''}"
    save_folder_name += f"{f'_{args.area_filter}' if args.area_filter else ''}"
    save_folder_name += f"{'_synced_detector' if args.synced_detector else ''}"
save_folder_name += f"_det{det_score_threshold}_{seg_class_id.name}"
save_dir = save_root_dir / save_folder_name / "processed"

suffix = folder_name_from_params(args.max_edge_length, not args.no_delta,
                                 not args.cartesian, args.no_sameframe, args.max_edge_distance_multiplier)
suffix += f"_steps{args.mpn_steps}"
suffix += f"_det{det_score_threshold}_scoremult"
suffix += f"_time{max_time_diff}"
suffix += f"_{reduced_score_threshold}{score_reduce_mode}"
suffix += f"{'_annotated' if annotated else ''}"
if args.edge_mlps_count != 3:
    suffix += f"_{args.edge_mlps_count}nodeMLPs"
if args.node_aggr_sections != 3:
    suffix += f"_{args.node_aggr_sections}aggsections"
suffix += f"{'_mini' if args.mini else ''}"
suffix += f"{f'_{args.area_filter}' if args.area_filter else ''}"
suffix += f"{'_synced_detector' if args.synced_detector else ''}"
suffix += f"_{seg_class_id.name}"

# Loading existing input graphs
det_graph_path = None
if args.no_delta or args.cartesian:
    det_graph_path = None
elif args.synced_detector:
    det_graph_path = None
elif args.max_edge_length == -1:
    if det_score_threshold == 0:
        det_graph_path = det_graphs_val_det0.get(seg_class_id, None)
    elif det_score_threshold == 0.2:
        det_graph_path = det_graphs_val_det02.get(seg_class_id, None)
    elif det_score_threshold == 0.3:
        det_graph_path = det_graphs_val_det03.get(seg_class_id, None)

models_folder = "ablation"
if args.mpn_steps != 4:
    models_folder += f"_steps{args.mpn_steps}"
elif args.no_sameframe:
    models_folder += "_no_sameframe"
elif args.no_delta and args.cartesian:
    models_folder += "_no_delta_cartesian"
elif args.no_delta:
    models_folder += "_no_delta"
elif args.cartesian:
    models_folder += "_cartesian"
else:
    models_folder += "_default"

if args.edge_mlps_count != 3:
    models_folder += f"_{args.edge_mlps_count}nodeMLPs"

if args.mini:
    models_folder += "_mini"

if args.area_filter:
    models_folder += f"_{args.area_filter}"

load_graphs = det_graph_path is not None
if load_graphs:
    assert det_graph_path
    dir_to_load = save_root_dir / det_graph_path  # val
    print(f"Loading {det_graph_path}")
else:
    print(f"Not loading an existing det graph")

# #######################################################

data_params_no_aug = {
    "seed": 124,
    "clip_len": 11 if args.max_edge_length == -1 else args.max_edge_length,
    "seg_class_id": seg_class_id,
    "annotated": annotated,
    "max_edge_length": int(args.max_edge_length),

    "include_dims": False,
    "deltas_only": not args.no_delta,
    "max_edge_distance": args.max_edge_distance_multiplier * max_edge_distances[seg_class_id],
    "online_only": False,
    "use_polar_attr": not args.cartesian,
    "link_past_tracks": False,
    "link_past_tracks_mode": None,
    "connect_intraframe": not args.no_sameframe,

    "bbox_drop_p": 0,
    "frame_drop_p": 0,
    "xz_std": 0,
    "theta_std": 0,
    "lwh_std": 0,

    "bbox_add_p": 0,
    "num_bboxes_to_always_add": 0,

    "det_score_threshold": det_score_threshold,
}

if dataset_to_eval == "nuscenes":
    if args.synced_detector:
        if args.mini:
            det_source = input_utils.CENTER_POINT_MINI
        elif args.area_filter == "boston":
            det_source = input_utils.CENTER_POINT_BOSTON
        elif args.area_filter == "singapore":
            det_source = input_utils.CENTER_POINT_SINGAPORE
    else:
        det_source = input_utils.CENTER_POINT

    mot_dataset = MOTDatasetNuScenes(work_dir=NUSCENES_WORK_DIR,
                                     det_source=det_source,
                                     seg_source=input_utils.MMDETECTION_CASCADE_NUIMAGES,
                                     params=NUSCENES_BEST_PARAMS,
                                     version="v1.0-test" if args.test else "v1.0-trainval")  # trainval, mini
    seq_names = mot_dataset.sequence_names(split)
elif dataset_to_eval == "kitti":
    mot_dataset = mot_kitti.MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
                                            det_source=input_utils.POINTGNN_T3,
                                            seg_source=input_utils.TRACKING_BEST,
                                            params=KITTI_BEST_PARAMS)
    seq_names = TRACK_VAL_SEQ

mot_dataset.params["do_not_init_model"] = True

ckpt_path = get_best_ckpt(Path(mot_dataset.work_dir) / models_folder, seg_class_id.name)
ckpt_folder_name = ckpt_path.parent.parent.parent.name
suffix += f"_{ckpt_folder_name[:14]}"
model_to_test = GraphTrackerOffline.load_from_checkpoint(ckpt_path,
                                                         hparams_file=str(ckpt_path.parent.parent / "hparams.yaml"))
cur_node_model = model_to_test.mpn_model.node_model
model_to_test.mpn_model.node_model = ContextualNodeModel(cur_node_model.flow_forward_model, 
                                                        cur_node_model.flow_frame_model, 
                                                        cur_node_model.flow_backward_model, 
                                                        cur_node_model.total_flow_model, 
                                                        cur_node_model.node_agg_mode, 
                                                        cur_node_model.attention_model, 
                                                        args.node_aggr_sections)
print(f"Loaded ckpt {str(ckpt_path)}")
target_class = seg_class_id.value  # CHECK THE CLASS

mot_dataset.reset(only_submission=True)
for seq_name in seq_names:
    sequence = mot_dataset.get_sequence(split, seq_name)

    print(f"{'NOT ' if not load_graphs else ''}loading saved graphs from disk")
    print(f"Processing the graph for {seq_name}")
    if load_graphs and (dir_to_load / "processed" / f"{seq_name}.pt").exists():
        data_list = SingleGraphDataset(dir_to_load, seq_name)
    else:
        if (save_dir / f"{seq_name}.pt").exists():
            data_list = SingleGraphDataset(save_dir, seq_name)
        else:
            data_list = graph_construction.from_dataset(mot_dataset, split, [seq_name], **data_params_no_aug)
            if not len(data_list):  # report nothing tracked for that sequence
                run_info = sequence.report_offline_tracking(target_class, {}, {}, suffix, annotated)
                gc.collect()
                continue
            data, slices = InMemoryDataset.collate(data_list)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save((data, slices), save_dir / f"{seq_name}.pt")

    print(f"Processing sequence {seq_name}")
    dataloader = DataLoader(data_list, shuffle=False, drop_last=False,
                            batch_size=batch_size, num_workers=NUM_WORKERS)

    batched_preds = Trainer(gpus=1, logger=False).predict(model_to_test, dataloaders=dataloader)
    batched_preds, _ = zip(*batched_preds)

    start_time = time.time()
    instance_matches = processing.map_predictions_to_detections(dataloader, batched_preds, max_time_diff)
    print(f"map_predictions_to_detections {time.time() - start_time:.2f}")

    start_time = time.time()
    instance_match_stats = offline_utils.reduce_match_scores(instance_matches, score_reduce_mode)
    instance_matches_sorted = sorted(instance_match_stats, key=lambda triplet: triplet[-1], reverse=True)
    print(f"reduce_match_scores + sort {time.time() - start_time:.2f}")

    start_time = time.time()
    instance_id_to_track_id, same_tracks_map, instance_id_to_track_score = processing.map_instances_to_tracks(instance_matches_sorted,
                                                                                                              reduced_score_threshold)

    print(f"map_instances_to_tracks {time.time() - start_time:.2f}")

    direct_mapping = offline_utils.streamline_mapping(same_tracks_map)

    start_time = time.time()
    track_ids_unique = set()
    instance_id_to_final_track_id: Dict[int, int] = {}
    for node_id, track_id in instance_id_to_track_id.items():
        track_id = direct_mapping.get(track_id, track_id)
        assert track_id not in direct_mapping
        instance_id_to_final_track_id[node_id] = track_id
        track_ids_unique.add(track_id)
    print(f"map to final tracks {time.time() - start_time:.2f}")

    print(f"{len(track_ids_unique)} unique tracks in total")

    run_info = sequence.report_offline_tracking(
        target_class, instance_id_to_final_track_id, instance_id_to_track_score, suffix, annotated)
    gc.collect()

mot_dataset.save_all_mot_results(run_info["mot_3d_file"])
os.write(1, f"Finished reporting tracking for {split}\n".encode())
