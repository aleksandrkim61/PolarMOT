import time
from typing import List, Iterable, Any, Dict
from collections import defaultdict
import argparse
import datetime
from pathlib import Path

import ujson as json

import dataset_classes.kitti.mot_kitti as mot_kitti
from dataset_classes.nuscenes.dataset import MOTDatasetNuScenes
from configs.params import TRACK_VAL_SEQ, build_params_dict, KITTI_BEST_PARAMS, NUSCENES_BEST_PARAMS, variant_name_from_params
from configs.local_variables import KITTI_WORK_DIR, SPLIT, NUSCENES_WORK_DIR
import inputs.utils as input_utils
from configs.inference_params import track_pred_thresholds_online, max_online_ages, track_pred_thresholds_offline, track_initial_mult_online
from configs.data_graph_build_params import max_edge_distances


def perform_tracking_full(dataset, target_sequences=[], sequences_to_exclude=[], print_debug_info=True):
    if len(target_sequences) == 0:
        target_sequences = dataset.sequence_names(SPLIT)

    total_frame_count = 0
    total_time = 0
    total_time_tracking = 0
    total_time_fusion = 0
    total_time_reporting = 0
    total_time_track_predicting = 0
    total_time_track_assigning = 0
    total_time_track_confidence_assigning = 0
    total_time_list_building = 0
    total_time_graph_construction = 0
    total_time_predict_latest_tracks = 0
    total_time_coordinate_reorder = 0
    run_info_all: Dict[str, Any] = defaultdict(float)

    for sequence_name in target_sequences:
        if len(sequences_to_exclude) > 0:
            if sequence_name in sequences_to_exclude:
                print(f'Skipped sequence {sequence_name}')
                continue

        print(f'Starting sequence: {sequence_name}')
        start_time = time.time()
        sequence = dataset.get_sequence(SPLIT, sequence_name)
        variant = variant_name_from_params(dataset.params)
        run_info = sequence.perform_tracking_for_eval()
        if "total_time_mot" not in run_info:
            continue

        total_time_sequence = time.time() - start_time
        total_time += total_time_sequence
        total_time_fusion += run_info["total_time_fusion"]
        total_time_tracking += run_info["total_time_mot"]
        total_time_reporting += run_info["total_time_reporting"]
        total_time_track_predicting += run_info["total_time_track_predicting"]
        total_time_track_assigning += run_info["total_time_track_assigning"]
        total_time_track_confidence_assigning += run_info["total_time_track_confidence_assigning"]
        total_time_list_building += run_info["total_time_list_building"]
        total_time_graph_construction += run_info["total_time_graph_construction"]
        total_time_predict_latest_tracks += run_info["total_time_predict_latest_tracks"]
        total_time_coordinate_reorder += run_info["total_time_coordinate_reorder"]
        total_frame_count += len(sequence.frame_names)

        for k, v in run_info.items():
            if not isinstance(v, float) and not isinstance(v, int):
                continue
            run_info_all[k] += v

        if print_debug_info:
            print(f'Sequence {sequence_name} took {total_time_sequence:.2f} sec, {total_time_sequence / 60.0 :.2f} min')
            print(
                f'Tracking took {run_info["total_time_mot"]:.2f} sec, {100 * run_info["total_time_mot"] / total_time_sequence:.2f}%')

    run_info["total_frame_count"] = total_frame_count
    run_info_all["total_frame_count"] = total_frame_count
    run_info_all["total_time"] = total_time
    if total_frame_count == 0:
        return variant, run_info

    dataset.save_all_mot_results(run_info["mot_3d_file"])

    if not print_debug_info:
        return variant, run_info, run_info_all

    # Overall variant stats
    # Timing
    print("\n")
    print(f'total_time_track_predicting  {total_time_track_predicting: .2f} sec, {(100 * total_time_track_predicting / total_time):.2f}%')
    print(f'total_time_track_assigning  {total_time_track_assigning: .2f} sec, {(100 * total_time_track_assigning / total_time):.2f}%')
    print(f'total_time_track_confidence_assigning  {total_time_track_confidence_assigning: .2f} sec, {(100 * total_time_track_confidence_assigning / total_time):.2f}%')
    print(f'total_time_list_building  {total_time_list_building: .2f} sec, {(100 * total_time_list_building / total_time):.2f}%')
    print(f'total_time_graph_construction  {total_time_graph_construction: .2f} sec, {(100 * total_time_graph_construction / total_time):.2f}%')
    print(f'total_time_predict_latest_tracks  {total_time_predict_latest_tracks: .2f} sec, {(100 * total_time_predict_latest_tracks / total_time):.2f}%')
    print(f'total_time_coordinate_reorder  {total_time_coordinate_reorder: .2f} sec, {(100 * total_time_coordinate_reorder / total_time):.2f}%')
    print(f'Tracking  {total_time_tracking: .2f} sec, {(100 * total_time_tracking / total_time):.2f}%')
    print(f'Reporting {total_time_reporting: .2f} sec, {(100 * total_time_reporting / total_time):.2f}%')
    print(
        f'Tracking-fusion framerate: {total_frame_count / (total_time_fusion + total_time_tracking):.2f} fps')
    print(f'Tracking-only framerate: {total_frame_count / total_time_tracking:.2f} fps')
    print(f'Total framerate: {total_frame_count / total_time:.2f} fps')
    print(f'Total time: {total_time} sec')
    print()
    print()
    for k, v in run_info_all.items():
        print(f"{k}: {v}")
    print()

    print(f"\n3D MOT saved in {run_info['mot_3d_file']}", end="\n\n")
    return variant, run_info, run_info_all


def track_dataset(dataset,
                  target_sequences: Iterable[str] = [],
                  sequences_to_exclude: Iterable[str] = []):
    start_time = time.time()
    variant, run_info, run_info_all = perform_tracking_full(dataset,
                                              target_sequences=target_sequences,
                                              sequences_to_exclude=sequences_to_exclude)
    records_file = Path(run_info["mot_3d_file"]) / "params.json"
    with open(records_file, 'w') as f:
        json.dump(dataset.params, f, indent=4)
    records_file = Path(run_info["mot_3d_file"]) / "run_info_all.json"
    with open(records_file, 'w') as f:
        json.dump(run_info_all, f, indent=4)
    print(f'Variant {variant} took {(time.time() - start_time) / 60.0:.2f} mins')
    return run_info


def adjust_params(params_mot: Dict[str, Any], args):
    params_mot["track_pred_thresholds"] = track_initial_mult_online  # track_pred_thresholds_online
    params_mot["track_initial_mult"] = track_initial_mult_online
    params_mot["max_edge_distances"] = max_edge_distances
    params_mot["max_edge_length"] = -1
    params_mot["include_dims"] = False
    params_mot["deltas_only"] = True
    params_mot["use_polar_attr"] = True
    params_mot["connect_intraframe"] = True

    if args.max_past_dets >= 0:
        params_mot["max_past_dets"] = tuple([args.max_past_dets] * 7)
    if args.det_score >= 0:
        params_mot["det_scores"] = tuple([args.det_score] * 7)
    params_mot["det_scores_to_report"] = tuple([args.det_score_to_report] * 7)
    params_mot["min_hits"] = tuple([args.min_hits] * 7)
    params_mot["max_frame_age"] = args.max_frame_age

    params_mot["online_only"] = not args.dense
    params_mot["link_past_tracks"] = not args.dense
    params_mot["link_past_tracks_mode"] = args.link_past_tracks_mode  # "consec"
    if params_mot["online_only"]:
        params_mot["pretrained_runs_folder"] = "ablation_default_online"
    else:
        params_mot["pretrained_runs_folder"] = "ablation_default"

    # # try models without further online training?
    # params_mot["pretrained_runs_folder"] = "ablation_default"
    # # needs to use offline track_pred thresholds
    # params_mot["track_pred_thresholds"] = track_pred_thresholds_offline

    params_mot["max_edge_distance_multiplier"] = args.max_edge_distance_multiplier

    run_name = f"{datetime.datetime.now().strftime('%H:%M')}_online"
    run_name += f"_{('link' + params_mot['link_past_tracks_mode']) if params_mot['link_past_tracks'] else 'dense'}"
    if args.track_score_thres >= 0:
        params_mot["track_pred_thresholds"] = defaultdict(lambda: args.track_score_thres)
        run_name += f"_track{args.track_score_thres}"
    if args.track_initial_mult >= 0:
        params_mot["track_initial_mult"] = defaultdict(lambda: args.track_initial_mult)
        run_name += f"_initialmult{args.track_initial_mult}"
    run_name += f"_h{args.min_hits}"
    run_name += f"_dist{args.max_edge_distance_multiplier}"
    run_name += f"_cull"
    run_name += f"_{args.run_name}"
    params_mot["run_name"] = run_name


def run_on_nuscenes(args):
    params_mot = NUSCENES_BEST_PARAMS
    adjust_params(params_mot, args)
    version = "v1.0-test" if args.test else "v1.0-trainval"
    mot_dataset = MOTDatasetNuScenes(work_dir=NUSCENES_WORK_DIR,
                                     det_source=input_utils.CENTER_POINT,
                                     seg_source=input_utils.MMDETECTION_CASCADE_NUIMAGES,
                                     params=params_mot,
                                     version=version)

    # if want to run on specific sequences only, add their str names here
    target_sequences: List[str] = []

    # if want to exclude specific sequences, add their str names here
    sequences_to_exclude: List[str] = []

    track_dataset(mot_dataset, target_sequences, sequences_to_exclude)
    mot_dataset.reset()


def run_on_kitti(args):
    params_mot = KITTI_BEST_PARAMS
    adjust_params(params_mot, args)
    # To reproduce "Ours (dagger)" results in Table II in the paper,
    # change det_source to input_utils.AB3DMOT and run on the VAL set
    mot_dataset = mot_kitti.MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
                                            det_source=input_utils.POINTGNN_T3,
                                            seg_source=input_utils.TRACKING_BEST,
                                            params=params_mot)

    # if want to run on specific sequences only, add their str names here
    target_sequences: List[str] = [] if args.test else TRACK_VAL_SEQ # TRACK_VAL_SEQ or []

    # if want to exclude specific sequences, add their str names here
    sequences_to_exclude: List[str] = []

    track_dataset(mot_dataset, target_sequences, sequences_to_exclude)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset on which to run, i.e. nuscenes or kitti")
    parser.add_argument("-run_name", type=str, default="", help="Suffix for the results folder")
    parser.add_argument("-det_score", type=float, default=-1, help="Minimum confidence for detections to consider, e.g. 0.2")
    parser.add_argument("-det_score_to_report", type=float, default=0, help="Minimum confidence for detections to report when no match was found, e.g. 0.2")
    parser.add_argument("-max_past_dets", type=int, default=-1, help="Maximum number of past frames kept in memory, e.g. 3, 5, -1 (take static variable defined in code)")
    parser.add_argument("-max_frame_age", type=int, default=20, help="Maximum number of frames in the clip for online tracking")
    parser.add_argument("-link_past_tracks_mode", type=str, default="last", help="Link past nodes mode")
    parser.add_argument("-min_hits", type=int, default=0, help="Minimum length of a track before reporting")
    parser.add_argument('-track_score_thres', type=float, default=-1, help="Link prediction threshold to constitute a match")
    parser.add_argument('-track_initial_mult', type=float, default=-1, help="Min link score multiplier")
    parser.add_argument('--dense', default=False, action='store_true')
    parser.add_argument("-max_edge_distance_multiplier", type=float, default=1,
                        help="Custom multiplier for default max connected edge distance")
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    if args.dataset.lower() == "nuscenes":
        run_on_nuscenes(args)
    elif args.dataset.lower() == "kitti":
        run_on_kitti(args)
