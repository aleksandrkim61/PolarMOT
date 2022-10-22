from typing import Mapping, Dict, Tuple, List, Iterable, Set, Deque
import time
from pathlib import Path
import datetime
from collections import defaultdict, deque
import statistics

import torch
from torch_geometric.data import Data
from numba.typed import List as nList
import numpy as np

from dataset_classes.mot_sequence import MOTSequence
from dataset_classes.utils import MAX_DETS_PER_FRAME


def map_predictions_to_detections(dataloader_, batched_preds,
                                  max_time_diff: int):
    """ Returns a dictionary with instance id matches and their scores

    :param max_time_diff: the maximum time difference between potential matches
    """
    instance_matches: Dict[Tuple[int, int], Deque[float]] = defaultdict(deque)  # (source_instance_i, target_instance_i): [list_of_scores]
    for batch_i, (batch, preds) in enumerate(zip(dataloader_, batched_preds)):
        preds = preds.cpu()
        preds_positive_mask = torch.ones_like(preds, dtype=bool).reshape((-1,))

        time_diff_allowed_mask = batch.edge_attr[:, -1] <= max_time_diff
        edges_to_take_mask = preds_positive_mask & time_diff_allowed_mask

        edge_index_to_take = batch.edge_index[:, edges_to_take_mask]
        edge_index_with_instance_ids = batch.instance_ids[edge_index_to_take].numpy().T
        preds_to_take = preds[edges_to_take_mask].numpy()

        for (start_instance, target_instance), match_score in zip(edge_index_with_instance_ids, preds_to_take):
            instance_matches[(start_instance, target_instance)].append(float(match_score))
    return instance_matches


def map_instances_to_tracks(instance_matches_sorted: Iterable[Tuple[int, int, float]], score_threshold: float):
    instance_id_to_track_id: Dict[int, int] = {}
    instance_id_to_track_score: Dict[int, float] = {}
    same_tracks_map: Dict[int, int] = {}
    # record which frames each track has already covered - to avoid overlap when joining tracks
    track_id_to_frame_indices: Dict[int, Set[int]] = defaultdict(set)
    
    track_id_latest = 0
    for start_instance, end_instance, score in instance_matches_sorted:        
        start_instance_frame_i = start_instance // MAX_DETS_PER_FRAME
        end_instance_frame_i = end_instance // MAX_DETS_PER_FRAME
        assert start_instance_frame_i != end_instance_frame_i, f"{start_instance} - {end_instance} with {score} in {start_instance_frame_i}"
        
        if start_instance not in instance_id_to_track_id and end_instance not in instance_id_to_track_id:
            # both are new - start a new track
            track_to_assign = track_id_latest
            track_id_latest += 1
        elif start_instance in instance_id_to_track_id and end_instance not in instance_id_to_track_id:
            # only one is already assigned
            track_to_assign = instance_id_to_track_id[start_instance]
            while track_to_assign in same_tracks_map:
                track_to_assign = same_tracks_map[track_to_assign]

            if end_instance_frame_i in track_id_to_frame_indices[track_to_assign]:
                # this track already has a detection in the that frame
                continue
        elif end_instance in instance_id_to_track_id and start_instance not in instance_id_to_track_id:
            track_to_assign = instance_id_to_track_id[end_instance]
            while track_to_assign in same_tracks_map:
                track_to_assign = same_tracks_map[track_to_assign]

            if start_instance_frame_i in track_id_to_frame_indices[track_to_assign]:
                # this track already has a detection in the that frame
                continue
        else:
            start_instance_track = instance_id_to_track_id[start_instance]
            end_instance_track = instance_id_to_track_id[end_instance]
            
            while start_instance_track in same_tracks_map:
                start_instance_track = same_tracks_map[start_instance_track]
            while end_instance_track in same_tracks_map:
                end_instance_track = same_tracks_map[end_instance_track]
            
            if start_instance_track == end_instance_track:
                continue
            
            if any(x in track_id_to_frame_indices[start_instance_track] 
                   for x in track_id_to_frame_indices[end_instance_track]):
                # these tracks cover at least one same frame - overlapping tracks remain independent
                continue

            if start_instance_track != end_instance_track:
                earlier_track = min(start_instance_track, end_instance_track)
                later_track = max(start_instance_track, end_instance_track)
                same_tracks_map[earlier_track] = later_track
                track_id_to_frame_indices[later_track].update(track_id_to_frame_indices[earlier_track])
                track_to_assign = later_track
            
        instance_id_to_track_id[start_instance] = track_to_assign
        instance_id_to_track_id[end_instance] = track_to_assign
        instance_id_to_track_score[end_instance] = score
        
        track_id_to_frame_indices[track_to_assign].add(start_instance_frame_i)
        track_id_to_frame_indices[track_to_assign].add(end_instance_frame_i)

        if score < score_threshold:
            print(f"Assigned {track_id_latest} tracks, combined {len(same_tracks_map)} of them")
            break

    return instance_id_to_track_id, same_tracks_map, instance_id_to_track_score
