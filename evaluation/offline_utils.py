from typing import Mapping, Dict, Tuple, List
import time
from pathlib import Path
import datetime
from collections import defaultdict, deque
import statistics


def reduce_match_scores(instance_matches: Mapping[Tuple, List], score_reduce_mode: str):
    if score_reduce_mode == "median":
        return [tup + (statistics.median(scores),) for tup, scores in instance_matches.items()]
    elif score_reduce_mode == "mean":
        return [tup + (statistics.mean(scores),) for tup, scores in instance_matches.items()]
    else:
        raise NotImplementedError(f"Score reduction '{score_reduce_mode}' is unknown")


def reduce_and_sort_match_scores_in_nested_dict(matches_dict: Mapping[int, Mapping[int, List]], score_reduce_mode: str):
    # Can be made more elegant with a function as an argument
    if score_reduce_mode == "median":
        return {start_node:
                deque(sorted([(target, statistics.median(scores))
                              for target, scores in target_scores_dict.items()],
                             key=lambda tup: tup[-1], reverse=True))
                for start_node, target_scores_dict in matches_dict.items()}
    elif score_reduce_mode == "mean":
        return {start_node:
                deque(sorted([(target, statistics.mean(scores))
                              for target, scores in target_scores_dict.items()],
                             key=lambda tup: tup[-1], reverse=True))
                for start_node, target_scores_dict in matches_dict.items()}
    else:
        raise NotImplementedError(f"Score reduction '{score_reduce_mode}' is unknown")


def streamline_mapping(same_tracks_map: Mapping[int, int]) -> Dict[int, int]:
    direct_mapping: Dict[int, int] = {}
    # later inserted assignments are used by earlier ones, so processing in reverse order should be faster
    for i, final_track_id in reversed(same_tracks_map.items()):
        while final_track_id in same_tracks_map:
            final_track_id = same_tracks_map[final_track_id]
        direct_mapping[i] = final_track_id
    return direct_mapping
