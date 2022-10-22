from __future__ import annotations
from typing import List, Iterable, Optional, Sequence, Tuple, Mapping, Union, Dict, Set
import time
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from scipy.spatial.distance import cdist

import dataset_classes.mot_dataset as mot_dataset
import dataset_classes.mot_sequence as mot_sequence
from dataset_classes.common import DatasetClassEnum
from tracking.utils_tracks import normalize_array_of_angles_inplace, fix_angles_from_difference_inplace, compute_rotation_around_y
from inputs.bbox import Bbox3d
import data.augmentation as augmentation
import data.utils as data_utils


def from_clip(bboxes_per_frame: Sequence[Sequence[Bbox3d]], timings: Dict[str, float], *,
              annotated: bool,
              max_edge_length: int,
              max_edge_distance: float,
              online_only: bool,
              link_past_tracks: bool,
              link_past_tracks_mode: str,
              deltas_only: bool,
              use_polar_attr: bool,
              connect_intraframe: bool = False,
              debug: bool = False,
              **kwargs) -> Optional[Data]:
    """ Generate a graph Data object from a clip of MOT frames
    Detections are only connected with detections in future frames - this avoids duplicating edges

    :param bboxes_per_frame: a collection of Bbox3d lists for each frame in the clip, size: (clip_length(num_dets))
    :param seg_class_id: which object class to consider, only car/ped/etc.
    :param annotated: whether to include labels = if labels are available in the supplied Bbox3d lists
    :param max_edge_length: how much maximum time can be  between nodes connected with an edge, 
        e.g. if max_edge_length=2, then detections are only connected up to 2 frames in the future
             if max_edge_length=-1, then detections are connected to all detections from other frames in the clip
    :param online_only: whether to construct the graps for an online use case - only connect to last frame in clip
        simulates past tracks that are still kept alive and last frame -> current frame connections
    :param link_past_tracks: if True, only connects last frame nodes to the closest nodes of the same track, ealier nodes are only connected
        consecutively to emulate correct tracking performed in earlier frames of the online mode
    :param debug: whether to print debug output - mostly shapes, defaults to False
    :param deltas_only: whether to divide edge features by time_diff, essentially giving deltas in poses as features
    :param max_edge_distance: maximum distance between nodes that will be connected with an edge    
                              negative value to not do any pruning and keep all edges

    :return: a graph Data object or None, if fewer than two frames have detections - nothing to connect
    """
    if link_past_tracks:
        assert online_only, "Cannot link_past_tracks in offline mode"
        assert link_past_tracks_mode in ("consec", "last")

    start_time = time.time()
    bboxes_flat = [bbox for bboxes_in_frame in bboxes_per_frame for bbox in bboxes_in_frame]
    if not bboxes_flat:
        return None
    total_bboxes_count = len(bboxes_flat)

    clip_len_actual = len(bboxes_per_frame)
    if max_edge_length < 0:
        max_edge_length = clip_len_actual - 1

    bboxes_count_per_frame = [len(bboxes) for bboxes in bboxes_per_frame]

    # In online mode, only connect to nodes from the last frame in the clip
    if online_only and bboxes_count_per_frame[-1] == 0:
        return None

    maximum_edge_count = sum((bboxes_count_per_frame[frame_i] * sum(bboxes_count_per_frame[frame_i+1:frame_i+1+max_edge_length])
                              for frame_i in range(clip_len_actual)))
    if maximum_edge_count == 0:
        return None  # skip clip with no detections

    # give flat sequential ids to all detections in the clip
    node_ids_per_frame: Tuple[torch.Tensor, ...] = tuple(torch.arange(sum(bboxes_count_per_frame[:i]),
                                                                      sum(bboxes_count_per_frame[:i]) + bboxes_count_per_frame[i])
                                                         for i in range(clip_len_actual))

    # Get bbox features or all frames here and get differences in the frames loops
    features_flat = np.vstack([bbox.gnn_features for bbox in bboxes_flat])
    # bring all angles to [0, 2PI]
    normalize_array_of_angles_inplace(features_flat[:, -1])

    instance_ids_flat = np.fromiter((bbox.instance_id for bbox in bboxes_flat), dtype=int)
    track_ids_flat = np.fromiter((bbox.gt_track_id for bbox in bboxes_flat), dtype=int)
    nodes_real_mask = track_ids_flat > -1

    start_attr_time = time.time()
    num_dense_connections = 0
    # build edges from nodes in each frame to all future frames
    # do not build edges backwards, no need to duplicate work in the future
    # TODO: can optimize this by using a stack - no need for contiguous memory
    edge_index_tensors: List[torch.Tensor] = []
    # holds distances or x_diffs depending on `use_polar_attr`
    edge_first_features_tensors: List[torch.Tensor] = []
    edge_second_features_tensors: List[torch.Tensor] = []  # holds polar_angles or z_diffs
    edge_orientation_diffs_tensors: List[torch.Tensor] = []
    edge_time_diffs_tensors: List[torch.Tensor] = []
    y_tensors: List[torch.Tensor] = []
    y_consecutive_tensors: List[torch.Tensor] = []

    track_ids_linked: Set[int] = set()
    # do not connect nodes from the last frame - nothing after them
    for frame_i in reversed(range(clip_len_actual - 1)):
        nodes_in_frame_i = node_ids_per_frame[frame_i]
        if not len(nodes_in_frame_i):
            continue

        # target nodes for each current node are nodes from future frames in the clip (up to max_edge_length)
        # Change here to only connect up-to a certain time range
        if not online_only:
            frame_indices_to_take = slice(frame_i+1, frame_i+1+max_edge_length, 1)
            # ids of future nodes to connect
            target_node_ids = torch.cat((node_ids_per_frame[frame_indices_to_take]))
            if len(target_node_ids) == 0:  # no connections from this frame
                continue
            # how many nodes from each future frame to connect - calculate time diff to each of them to use as a feature
            targets_in_future_count = bboxes_count_per_frame[frame_indices_to_take]
            time_diff = torch.cat([torch.full((count, 1), i+1)
                                   for i, count in enumerate(targets_in_future_count) if count > 0])
        else:
            target_node_ids = node_ids_per_frame[-1]
            if len(target_node_ids) == 0:  # no connections from this frame
                continue
            time_diff = torch.full((len(target_node_ids), 1), clip_len_actual - 1 - frame_i)
        time_diff = time_diff.reshape(-1,)
        assert len(time_diff) == len(target_node_ids)

        source_features = features_flat[nodes_in_frame_i].reshape(-1, 3)
        target_features = features_flat[target_node_ids].reshape(-1, 3)
        _pairwise_attributes = calculate_pairwise_attributes_from_features(
            source_features, target_features, use_polar_attr, timings)
        # dist or x_diffs    polar_angle or z_diff
        edge_first_features, edge_second_features, orientation_diffs, distances = _pairwise_attributes

        target_track_ids = torch.tensor(track_ids_flat[target_node_ids]).reshape(-1,)

        # connect each node in the current frame
        for node_i, node_id in enumerate(nodes_in_frame_i):
            source_track_id = track_ids_flat[node_id]
            if link_past_tracks and source_track_id in track_ids_linked:
                # For online, only the last detection in each track needs to be connected to the latest frame, others need to be connected consequtively
                continue

            # Can threshold by distance here, not accumulate a lot of extra space
            # take the whole row - it will be distances to all its targets

            if max_edge_distance < 0:
                indices_to_keep = torch.ones((len(target_node_ids),))
            else:
                distances_current = distances[node_i] / time_diff  # distance per timestep/frame
                indices_to_keep = (distances_current <= max_edge_distance)
                if not any(indices_to_keep):
                    continue

            if source_track_id != -1:
                track_ids_linked.add(source_track_id)

            edge_first_features_tensors.append(edge_first_features[node_i][indices_to_keep])
            edge_second_features_tensors.append(edge_second_features[node_i][indices_to_keep])
            edge_orientation_diffs_tensors.append(orientation_diffs[node_i][indices_to_keep])
            edge_time_diffs_tensors.append(time_diff[indices_to_keep])

            # each current frame node connects to the same targets from other frames
            target_node_ids_current = target_node_ids[indices_to_keep]
            source_node_ids = torch.full_like(target_node_ids_current, node_id)
            edge_index_tensors.append(torch.vstack((source_node_ids, target_node_ids_current)))
            num_dense_connections += len(target_node_ids_current)

            if annotated:
                target_track_ids_kept = target_track_ids[indices_to_keep]
                if source_track_id == -1:
                    same_track = torch.zeros_like(target_track_ids_kept)
                else:
                    same_track = source_track_id == target_track_ids_kept
                y_tensors.append(same_track)
                # only keep the first True index
                same_track_consec_padded = torch.zeros((len(same_track) + 1,), dtype=bool)
                # padded array to allow setting index -1 if a true values was not found
                same_track_consec_padded[data_utils.first_true_index_or_minus_one(same_track.numpy())] = True
                y_consecutive_tensors.append(same_track_consec_padded[:-1])
    timings["time_connect_dense_edges"] += time.time() - start_attr_time

    start_link_past_tracks = time.time()
    # link tracked nodes consecutively
    if link_past_tracks:
        for source_frame_i in range(clip_len_actual - 2):  # do not connect nodes from the 2nd to last frame
            nodes_in_frame_i = node_ids_per_frame[source_frame_i]
            if not len(nodes_in_frame_i):
                continue

            frame_indices_to_take = slice(source_frame_i+1, min(source_frame_i +
                                          1+max_edge_length, clip_len_actual-1), 1)
            # ids of future nodes to connect
            target_node_ids = torch.cat((node_ids_per_frame[frame_indices_to_take]))
            if len(target_node_ids) == 0:  # no connections from this frame
                continue
            # how many nodes from each future frame to connect - calculate time diff to each of them to use as a feature
            targets_in_future_count = bboxes_count_per_frame[frame_indices_to_take]
            time_diff = torch.cat([torch.full((count, 1), i+1)
                                   for i, count in enumerate(targets_in_future_count) if count > 0])
            time_diff = time_diff.reshape(-1,)
            assert len(time_diff) == len(target_node_ids)

            source_features = features_flat[nodes_in_frame_i].reshape(-1, 3)
            target_features = features_flat[target_node_ids].reshape(-1, 3)
            _pairwise_attributes = calculate_pairwise_attributes_from_features(
                source_features, target_features, use_polar_attr, timings)
            edge_first_features, edge_second_features, orientation_diffs, distances = _pairwise_attributes

            target_track_ids = torch.tensor(track_ids_flat[target_node_ids]).reshape(-1,)

            # connect each node in the current frame
            for node_i, node_id in enumerate(nodes_in_frame_i):
                source_track_id = track_ids_flat[node_id]
                if source_track_id == -1:
                    continue

                same_track = source_track_id == target_track_ids
                if link_past_tracks_mode == "consec":
                    # only keep the first True index
                    target_index_to_link = data_utils.first_true_index_or_minus_one(same_track.numpy())
                elif link_past_tracks_mode == "last":
                    # only connect with the last det of the track - dense connections for the node that can get matched
                    target_index_to_link = data_utils.last_true_index_or_minus_one(same_track.numpy())
                if target_index_to_link == -1:  # no target with the same track_id
                    continue
                # TODO: Add a random link (or two) for online to simulate wrong connections. Later edge_drop augmentations will help

                edge_first_features_tensors.append(
                    edge_first_features[node_i][target_index_to_link].reshape(1,))
                edge_second_features_tensors.append(
                    edge_second_features[node_i][target_index_to_link].reshape(1,))
                edge_orientation_diffs_tensors.append(
                    orientation_diffs[node_i][target_index_to_link].reshape(1,))
                edge_time_diffs_tensors.append(time_diff[target_index_to_link].reshape(1,))

                edge_index_tensors.append(torch.tensor(
                    [node_id, target_node_ids[target_index_to_link]]).reshape(2, 1))

                if annotated:
                    y_tensors.append(torch.ones((1,)))
                    y_consecutive_tensors.append(torch.ones((1,)))
    timings["time_link_past_tracks"] += time.time() - start_link_past_tracks

    if not edge_index_tensors:
        return None

    start_intraframe = time.time()
    same_frame_edge_index_tensors: List[torch.Tensor] = []
    same_frame_edge_first_features_tensors: List[torch.Tensor] = []
    same_frame_edge_second_features_tensors: List[torch.Tensor] = []
    same_frame_edge_orientation_diffs_tensors: List[torch.Tensor] = []
    if connect_intraframe:  # connect nodes in the same frame
        # intra frame edges are in all frames, incl the last one
        for frame_i in reversed(range(clip_len_actual)):
            nodes_in_frame_i = node_ids_per_frame[frame_i]
            if not len(nodes_in_frame_i):
                continue

            source_features = features_flat[nodes_in_frame_i].reshape(-1, 3)
            _pairwise_attributes = calculate_pairwise_attributes_from_features(
                source_features, source_features, use_polar_attr, timings)
            edge_first_features, edge_second_features, orientation_diffs, distances = _pairwise_attributes

            # connect each node in the current frame to all subsequent nodes - ordered by node_id
            for node_i, node_id in enumerate(nodes_in_frame_i[:-1]):
                if max_edge_distance < 0:
                    indices_to_keep = torch.ones((len(nodes_in_frame_i),), dtype=bool)
                else:
                    indices_to_keep = (distances[node_i] <= (max_edge_distance * 2))  # 2x distance threshold
                    if not any(indices_to_keep):
                        continue
                indices_to_keep[:(node_i+1)] = False  # do not connect previous nodes

                same_frame_edge_first_features_tensors.append(edge_first_features[node_i][indices_to_keep])
                same_frame_edge_second_features_tensors.append(edge_second_features[node_i][indices_to_keep])
                same_frame_edge_orientation_diffs_tensors.append(orientation_diffs[node_i][indices_to_keep])

                # each current frame node connects to the same targets from other frames
                target_node_ids_current = nodes_in_frame_i[indices_to_keep]
                source_node_ids = torch.full_like(target_node_ids_current, node_id)
                same_frame_edge_index_tensors.append(torch.vstack((source_node_ids, target_node_ids_current)))
    timings["time_connect_intraframe"] += time.time() - start_intraframe

    ################################################################
    # Stack tensors, normalize by time, no more changes to the graph
    timings["time_attr_compute"] += time.time() - start_attr_time

    edge_index = torch.hstack(edge_index_tensors)
    assert torch.all(edge_index[0] < edge_index[1]), "Some edges to the past were built"
    edge_features = torch.vstack((torch.cat(edge_first_features_tensors),
                                 torch.cat(edge_second_features_tensors),
                                 torch.cat(edge_orientation_diffs_tensors))).T
    edge_time_diff = torch.cat(edge_time_diffs_tensors).reshape(-1, 1)
    assert len(
        edge_time_diff) == edge_index.shape[1], f"edge_time_diff {edge_time_diff.shape}, edge_index {edge_index.shape}"
    assert len(edge_features) == len(
        edge_time_diff), f"edge_features {edge_features.shape}, edge_time_diff {edge_time_diff.shape}"
    assert not is_undirected(edge_index, num_nodes=total_bboxes_count)

    if deltas_only:
        edge_features /= edge_time_diff  # features per time step, deltas only

    edge_attr = torch.hstack((edge_features, edge_time_diff))
    assert edge_attr.shape == (edge_index.shape[1], 4), f"edge_attr {edge_attr.shape}"

    if not same_frame_edge_index_tensors:
        same_frame_edge_index = torch.empty((2, 0))
        same_frame_edge_features = torch.empty((0, 3))
        same_frame_edge_time_diff = torch.empty((0, 1))
    else:
        same_frame_edge_index = torch.hstack(same_frame_edge_index_tensors)
        assert torch.all(same_frame_edge_index[0] < same_frame_edge_index[1]), "Wrong edges were built"
        same_frame_edge_features = torch.vstack((torch.cat(same_frame_edge_first_features_tensors),
                                                 torch.cat(same_frame_edge_second_features_tensors),
                                                 torch.cat(same_frame_edge_orientation_diffs_tensors)
                                                 )).T
        same_frame_edge_time_diff = torch.zeros((len(same_frame_edge_features), 1))
    assert len(
        same_frame_edge_time_diff) == same_frame_edge_index.shape[1], f"{same_frame_edge_time_diff.shape}, {same_frame_edge_index.shape}"
    assert len(same_frame_edge_features) == len(
        same_frame_edge_time_diff), f"{same_frame_edge_features.shape}, {same_frame_edge_time_diff.shape}"

    same_frame_edge_attr = torch.hstack((same_frame_edge_features, same_frame_edge_time_diff))
    assert same_frame_edge_attr.shape == (same_frame_edge_index.shape[1], 4), f"{edge_attr.shape}"

    if debug:
        print("bboxes_count_per_frame", bboxes_count_per_frame)
        print("node_ids_per_frame", node_ids_per_frame)
        print("edge_index", edge_index.shape)
        print(edge_index)
        print("edge_attr", edge_attr.shape)
        print(edge_attr)

    data = Data(num_nodes=total_bboxes_count, instance_ids=torch.from_numpy(instance_ids_flat),
                edge_index=edge_index, edge_attr=edge_attr.float())
    if connect_intraframe:
        data.same_frame_edge_index = same_frame_edge_index
        data.same_frame_edge_attr = same_frame_edge_attr.float()
    if online_only:
        dense_connections_mask = torch.zeros((edge_index.shape[1], ), dtype=bool)
        dense_connections_mask[:num_dense_connections] = True
        data.dense_connections_mask = dense_connections_mask
    if annotated:
        data.nodes_real_mask = torch.from_numpy(nodes_real_mask)
        data.y = torch.cat(y_tensors).float()
        data.y_consecutive = torch.cat(y_consecutive_tensors).float()
        assert len(
            data.y) == edge_index.shape[1], f"Not all edges have a label { len(data.y)} != {edge_index.shape[1]}"
        assert len(
            data.y_consecutive) == edge_index.shape[1], f"Not all edges have a consec label { len(data.y_consecutive)} != {edge_index.shape[1]}"
    timings["time_building_clip"] += time.time() - start_time
    return data


def calculate_pairwise_attributes_from_features(source_features, target_features, use_polar_attr: bool, timings: Dict[str, float]):
    start_time = time.time()
    pairwise_matrix_shape = (len(source_features), len(target_features))

    # get differences in bbox orientations for each pair
    # matrix [len(target) x len(source)] direction does not matter as long as consistent
    # Incompatible with earlier models - diff used to be in [-PI/2, PI/2]
    orientation_diffs = np.subtract.outer(source_features[:, -1], target_features[:, -1])
    fix_angles_from_difference_inplace(orientation_diffs)  # all in [-PI, PI]
    orientation_diffs *= -1  # -1 to make it target -> source - consistent with polar
    assert orientation_diffs.shape == pairwise_matrix_shape, f"{orientation_diffs.shape}"

    x_diffs = np.subtract.outer(source_features[:, 0], target_features[:, 0])
    x_diffs *= -1  # -1 to make it target -> source, important for polar angles
    assert x_diffs.shape == pairwise_matrix_shape, f"{x_diffs.shape}"

    z_diffs = np.subtract.outer(source_features[:, 1], target_features[:, 1])
    z_diffs *= -1  # -1 to make it target -> source
    assert z_diffs.shape == pairwise_matrix_shape, f"{z_diffs.shape}"

    if use_polar_attr:
        # get pair-specific polar angles - angle difference between source's current orientation and the target's center
        # computed in two steps:
        # 1) calculate the rotation around Y of the vector between source's and target's centers, i.e. the difference vector
        orientation_of_difference_vectors = compute_rotation_around_y(x_diffs, z_diffs)
        normalize_array_of_angles_inplace(orientation_of_difference_vectors)  # bring all angles to [0, 2PI]
        # 2) calculate the difference between that angle and the source's current orientation
        #    the result is the needed polar angle, where the pole is the source's center and its current heading vector is the polar axis
        polar_angles = source_features[:, -1].reshape(-1, 1) - orientation_of_difference_vectors
        fix_angles_from_difference_inplace(polar_angles)
        assert polar_angles.shape == pairwise_matrix_shape, f"{polar_angles.shape}"
        polar_angles = torch.from_numpy(polar_angles)

    x_diffs = torch.from_numpy(x_diffs)
    z_diffs = torch.from_numpy(z_diffs)
    orientation_diffs = torch.from_numpy(orientation_diffs)
    distances = np.linalg.norm(np.stack((x_diffs, z_diffs)), axis=0)
    assert distances.shape == pairwise_matrix_shape, f"{distances.shape}"
    distances = torch.from_numpy(distances)

    timings["time_calculate_pairwise_attributes_from_features"] += time.time() - start_time

    # TODO: stack features (dist, polar, orientation) to a single array NxMx3 and avoid separate lists
    if use_polar_attr:
        return distances, polar_angles, orientation_diffs, distances
    else:
        return x_diffs, z_diffs, orientation_diffs, distances
    # return x_diffs, z_diffs, distances, orientation_diffs, polar_angles


def from_sequence(sequence: mot_sequence.MOTSequence, clip_len: int, seg_class_id: DatasetClassEnum, timings: Dict[str, float], *,
                  annotated: bool,
                  bbox_drop_p: float, frame_drop_p: float,
                  xz_std: float, theta_std: float, lwh_std: float,
                  bbox_add_p: float, num_bboxes_to_always_add: int,
                  det_score_threshold: float = 0,
                  starting_frame: int = 0,
                  debug: bool = False, **kwargs) -> List[Data]:
    start_time = time.time()
    frames_all = [sequence.get_frame(frame_name) for frame_name in sequence.frame_names]
    if annotated:
        bboxes_per_frame_all = [[bbox_3d for bbox_3d in frame.bbox_3d_annotations(world=True)
                                if bbox_3d.seg_class_id == seg_class_id.value] for frame in frames_all]
    else:
        bboxes_per_frame_all = [[bbox_3d for bbox_3d in frame.bboxes_3d_world
                                 if bbox_3d.seg_class_id == seg_class_id.value and bbox_3d.confidence >= det_score_threshold]
                                for frame in frames_all]
    timings["time_bbox_parsing"] += time.time() - start_time

    data_list = []
    for start_frame_i in range(starting_frame, len(sequence.frame_names)):
        frames_to_take = slice(start_frame_i, start_frame_i + clip_len)
        if start_frame_i % 200 == 0:
            print(f"Processing frame {start_frame_i}")
        if debug:
            print(f"Processing {sequence.frame_names[frames_to_take]} ")

        bboxes_per_frame = bboxes_per_frame_all[frames_to_take]

        start_time = time.time()
        bboxes_per_frame = augmentation.add_bboxes_to_clip(bboxes_per_frame, seg_class_id.value,
                                                           bbox_add_p=bbox_add_p,
                                                           num_bboxes_to_always_add=num_bboxes_to_always_add)
        timings["time_add_bboxes"] += time.time() - start_time

        start_time = time.time()
        bboxes_per_frame = augmentation.drop_frames_from_clip(bboxes_per_frame, frame_drop_p=frame_drop_p)
        bboxes_per_frame = augmentation.drop_bboxes_from_clip(bboxes_per_frame, bbox_drop_p=bbox_drop_p)
        timings["time_drop_bboxes_frames"] += time.time() - start_time

        start_time = time.time()
        bboxes_per_frame = augmentation.jitter_bboxes_in_clip(
            bboxes_per_frame, xz_std=xz_std, theta_std=theta_std, lwh_std=lwh_std)
        timings["time_jitter_bboxes"] += time.time() - start_time

        data = from_clip(bboxes_per_frame, timings, annotated=annotated, debug=debug, **kwargs)
        if data is not None:
            data.start_frame_i = start_frame_i
            data_list.append(data)
    return data_list


def from_dataset(mot_dataset: mot_dataset.MOTDataset, split: str, target_sequences: Optional[List[str]] = None, *,
                 clip_len: int, seg_class_id: DatasetClassEnum, area_filter: str = "", **kwargs) -> List[Data]:
    timings: Dict[str, float] = defaultdict(float)
    start_time = time.time()

    if not target_sequences:
        target_sequences = mot_dataset.sequence_names(split)
    assert target_sequences  # to calm mypy down
    original_len = len(target_sequences)

    if area_filter:
        target_sequences = [scene_name for scene_name in target_sequences
                            if scene_name in mot_dataset.scene_names_for_area(area_filter)]
        print(f"{len(target_sequences)}/{original_len} scenes from {area_filter}")
    else:
        print(f"Not filtered {original_len} scenes")

    data_all = []
    for sequence_name in target_sequences:
        print(f"Processing sequence {sequence_name}")
        data_all.extend(from_sequence(mot_dataset.get_sequence(split, sequence_name),
                                      clip_len, seg_class_id, timings, **kwargs))

    print(
        f"Building data objects for {split} from {len(target_sequences)} sequences for length {clip_len}, class {seg_class_id} " +
        f"took {(time.time() - start_time) / 60.0 :.2f} minutes"
    )
    print(f"{len(data_all)} data objects in total")

    print(f"time_bbox_parsing        {(timings['time_bbox_parsing'] / 60) :>5.3f} minutes")
    print(f"time_building_clip      {(timings['time_building_clip'] / 60) :>5.3f} minutes")
    print(f"time_attr_compute       {(timings['time_attr_compute'] / 60) :>5.3f} minutes")
    print(f"time_drop_bboxes_frames {(timings['time_drop_bboxes_frames'] / 60) :>5.3f} minutes")
    print(f"time_jitter_bboxes      {(timings['time_jitter_bboxes'] / 60) :>5.3f} minutes")
    print(f"time_add_bboxes         {(timings['time_add_bboxes'] / 60) :>5.3f} minutes")
    return data_all
