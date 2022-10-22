from typing import List, Iterable, Optional, Sequence
import time
import math
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import Data

from inputs.bbox import Bbox3d
from dataset_classes.common import DatasetClassEnum

# TODO: change to 6-8m or something, 20 is way too far
# An well-planed urban road is ~14.5meters, including 2 driving lanes, 1 bus, 1 parking, 1 divisor, 1 bike
# Scenarios on KITTI and NuScenes are dense urban roads without full infrastructure, so probably ~8-12m
XZ_MIN_SPREAD = 14.0  # to create fake boxes in a minimum range
MIN_EDGES = 2
MIN_NODES = 2


def drop_frames_from_clip(bboxes_per_frame: List[List[Bbox3d]], *,
                          frame_drop_p: float) -> List[List[Bbox3d]]:
    """ Drop whole frames from a clip

    :param bboxes_per_frame: bboxes for each frame in the clip
    :param frame_drop_p: probability of dropping each frame from the given clip
    :return: a list with some bboxes dropped out
    """
    if not bboxes_per_frame:
        return [[]]
    if frame_drop_p == 0:
        return bboxes_per_frame

    # Sensor fault simulation - each drop decision is independent
    return [frame_bboxes if np.random.uniform() > frame_drop_p else [] for frame_bboxes in bboxes_per_frame]


def drop_bboxes(bboxes: List[Bbox3d], *,
                bbox_drop_p: float) -> List[Bbox3d]:
    """ Drop bboxes from a frame

    :param bboxes: a list of original bboxes
    :param bbox_drop_p: probability of dropping each bbox from the given clip
    :return: a list with some bboxes dropped out
    """
    if not bboxes:
        return []
    if bbox_drop_p == 0:
        return bboxes

    # Sensor fault simulation - each drop decision is independent
    return [b for b in bboxes if np.random.uniform() > bbox_drop_p]

class DropNodes(object):
    """ Drop nodes from a constructed data object,
    also drops corresponding attributes and labels if present.

    :param drop_p: probability of dropping each node
    """
    def __init__(self, drop_p: float):
        self.drop_p = drop_p

    def __call__(self, data: Data) -> Data:
        if self.drop_p == 0 or data.num_nodes <= MIN_NODES:
            return data

        node_survival_prob = np.random.uniform(size=data.num_nodes)
        nodes_to_keep = node_survival_prob > self.drop_p
        node_ids_to_keep = nodes_to_keep.nonzero()[0]
        nodes_kept_count = len(node_ids_to_keep)
        if nodes_kept_count <= MIN_NODES:
            return data

        edges_to_keep = np.isin(data.edge_index.numpy(), node_ids_to_keep).all(0)  # if source and target nodes are kept
        data.edge_index = data.edge_index[:, edges_to_keep]
        data.edge_attr = data.edge_attr[edges_to_keep]

        if data.y is not None:
            data.y = data.y[edges_to_keep]
            data.y_consecutive = data.y_consecutive[edges_to_keep]
        if hasattr(data, "dense_connections_mask"):
            data.dense_connections_mask = data.dense_connections_mask[edges_to_keep]
        if hasattr(data, "same_frame_edge_index"):
            edges_to_keep = np.isin(data.same_frame_edge_index.numpy(), node_ids_to_keep).all(0)
            data.same_frame_edge_index = data.same_frame_edge_index[:, edges_to_keep]
            data.same_frame_edge_attr = data.same_frame_edge_attr[edges_to_keep]
        # Do not need to change nodes, can just delete edges - saves time but increases vRAM requirements because empty nodes are maintained

        # data.num_nodes = nodes_kept_count
        # if hasattr(data, "instance_ids"):
        #     data.instance_ids = data.instance_ids[nodes_to_keep]
        # if hasattr(data, "nodes_real_mask"):
        #     data.nodes_real_mask = data.nodes_real_mask[nodes_to_keep]

        # # TODO: Rework this - takes a long time 100ms
        # # maybe construct the whole array that will be subtracted in one go somehow
        # # shift edge indices down to account for dropped nodes and keep indices sequential
        # node_ids_to_drop = (~nodes_to_keep).nonzero()[0]
        # for i in sorted(node_ids_to_drop, reverse=True):
        #     data.edge_index[data.edge_index > i] -= 1
        #     if hasattr(data, "same_frame_edge_index"):
        #         data.same_frame_edge_index[data.same_frame_edge_index > i] -= 1
        return data


class DropEdges(object):
    """ Drop edges from a constructed data object,
    also drops corresponding attributes and labels if present.

    :param drop_p: probability of dropping each edge
    """
    def __init__(self, drop_p: float, online_past_edges: bool):
        self.drop_p = drop_p
        self.online_past_edges = online_past_edges

    def __call__(self, data: Data) -> Data:
        if self.drop_p == 0 or data.num_edges <= MIN_EDGES:
            return data

        if self.online_past_edges:
            edges_to_keep = torch.ones((data.edge_index.shape[1],), dtype=bool)
            num_past_edges = data.edge_index.shape[1] - data.dense_connections_mask.sum()
            past_edge_survival_prob = np.random.uniform(size=num_past_edges)  # only possibly drop past edges
            past_edges_to_keep = past_edge_survival_prob > self.drop_p
            # keep all dense edge, drop non-surviving past ones
            edges_to_keep[~data.dense_connections_mask][~past_edges_to_keep] = False
            edges_kept_count = edges_to_keep.sum()
        else:
            edge_survival_prob = np.random.uniform(size=data.edge_index.shape[1])
            edges_to_keep = edge_survival_prob > self.drop_p
            edges_kept_count = edges_to_keep.sum()
        if edges_kept_count <= MIN_EDGES:
            return data

        data.edge_index = data.edge_index[:, edges_to_keep]
        data.edge_attr = data.edge_attr[edges_to_keep]
        if data.y is not None:
            data.y = data.y[edges_to_keep]
            data.y_consecutive = data.y_consecutive[edges_to_keep]
        if hasattr(data, "dense_connections_mask"):
            data.dense_connections_mask = data.dense_connections_mask[edges_to_keep]
        if hasattr(data, "same_frame_edge_index"):
            edge_survival_prob = np.random.uniform(size=data.same_frame_edge_index.shape[1])
            edges_to_keep = edge_survival_prob > self.drop_p
            data.same_frame_edge_index = data.same_frame_edge_index[:, edges_to_keep]
            data.same_frame_edge_attr = data.same_frame_edge_attr[edges_to_keep]
        return data


def generate_gauss_offsets(std: float, n: int):
    return np.random.normal(scale=std, size=(n, 1)) if std > 0 else np.zeros((n, 1))


def jitter_bboxes(bboxes: Sequence[Bbox3d], *,
                  xz_std: float,  # 0.4
                  theta_std: float,  # 0.17 ~5 degrees
                  lwh_std: float,  # 0.3
                  ) -> List[Bbox3d]:
    """ Jitter each of the given bboxes.
    Offsets are taken from a zero-centered Gaussian with given standard deviation (separate for box centroid and dimensions)

    :param xz_std: standard deviation for the probability from which centroid offsets will be sampled
    :param theta_std: standard deviation for the probability from which orientation angle offsets will be sampled
    :param lwh_std: standard deviation for the probability from which box size offsets will be sampled
    Note: 68.3% are within std, 95.5% are within 2*std and 99.7% are within 3*std
    :return: a list with jittered bboxes
    """
    if not bboxes:
        return []

    N = len(bboxes)

    x_offsets = generate_gauss_offsets(xz_std, N)
    z_offsets = generate_gauss_offsets(xz_std, N)
    theta_offsets = generate_gauss_offsets(theta_std, N)
    # do not change elevation or dimensions, they are ignored
    y_offsets = np.zeros((N, 1))  
    lwh_offsets = np.zeros((N, 3))
    offsets = np.hstack((x_offsets, y_offsets, z_offsets, theta_offsets, lwh_offsets))

    # (x y z rotation-around-y l(x) w(z) h(y))
    current_coordinates = np.vstack([bbox.kf_coordinates for bbox in bboxes])
    assert current_coordinates.shape == (N, 7)
    assert current_coordinates.shape == offsets.shape, f"coord {current_coordinates.shape}, offsets {offsets.shape}"

    augmented_coordinates = current_coordinates + offsets
    # bboxes_augmented = [Bbox3d(aug_coord, bbox.gt_track_id, bbox.confidence, bbox.obs_angle, bbox.seg_class_id,
    #                            bbox.velocity, bbox.info, bbox._bbox_2d_in_cam, instance_id=bbox.instance_id)
    #                     for bbox, aug_coord in zip(bboxes, augmented_coordinates)]

    # TODO: Check KITTI training, should have helped actually, or maybe not - this is what Aljosa was wondering about
    bboxes_augmented = deepcopy(bboxes)
    for bbox, aug_coord in zip(bboxes_augmented, augmented_coordinates):
        bbox.kf_coordinates = aug_coord
    return list(bboxes_augmented)


class JitterEdgeAttr(object):
    def __init__(self, *, dist_x_std: float, polar_z_std: float, theta_std: float):
        self.dist_x_std = dist_x_std
        self.polar_z_std = polar_z_std
        self.theta_std = theta_std

    def __call__(self, data: Data) -> Data:
        if self.dist_x_std == 0 and self.polar_z_std == 0 and self.theta_std == 0:
            return data

        N = len(data.edge_attr)
        dist_x_offsets = generate_gauss_offsets(self.dist_x_std, N)
        polar_z_offsets = generate_gauss_offsets(self.polar_z_std, N)
        theta_offsets = generate_gauss_offsets(self.theta_std, N)
        offsets = np.hstack((dist_x_offsets, polar_z_offsets, theta_offsets))
        assert offsets.shape == (N, data.edge_attr.shape[1] - 1)
        data.edge_attr[:, :-1] += offsets

        if hasattr(data, "same_frame_edge_index"):
            N = len(data.same_frame_edge_attr)
            dist_x_offsets = generate_gauss_offsets(self.dist_x_std, N)
            polar_z_offsets = generate_gauss_offsets(self.polar_z_std, N)
            theta_offsets = generate_gauss_offsets(self.theta_std, N)
            offsets = np.hstack((dist_x_offsets, polar_z_offsets, theta_offsets))
            assert offsets.shape == (N, data.same_frame_edge_attr.shape[1] - 1)
            data.same_frame_edge_attr[:, :-1] += offsets
        return data


def add_bboxes(bboxes: List[Bbox3d], seg_class_id: int, *,
               bbox_add_p: float,
               num_bboxes_to_always_add: int) -> List[Bbox3d]:
    """ Add extra fake boxes to the frame.
    The number of added boxes is the sum of the fraction of existing ones and `num_bboxes_to_always_add`.
    Place boxes in uniformly sampled coordinates taken inside the range of [min, max] of real coordinates

    :param bbox_add_p: probability of adding an extra bbox for each existing one
    :param num_bboxes_to_always_add: number of extra boxes to add regardless of the fraction
    :return: a list containing real and fake bboxes
    """
    N = len(bboxes)
    num_to_add = sum(np.random.uniform(size=N) < bbox_add_p) + num_bboxes_to_always_add
    if not num_to_add:
        return bboxes

    if N == 0:
        min_coords = np.array([0, 0, 0, -1.5, 1, 1, 1])
        max_coords = np.array([0, 0, 0,  1.5, 4, 4, 3])
    else:
        current_coordinates = np.vstack([bbox.kf_coordinates for bbox in bboxes])
        min_coords = current_coordinates.min(axis=0)
        max_coords = current_coordinates.max(axis=0)

    # Make sure min/max spread is a minimum of `XZ_MIN_SPREAD`
    center_xz_coords = (min_coords[(0, 2), ] + max_coords[(0, 2), ]) / 2
    assert len(center_xz_coords) == 2

    min_coords[0] = min(min_coords[0], center_xz_coords[0] - XZ_MIN_SPREAD / 2)
    max_coords[0] = max(max_coords[0], center_xz_coords[0] + XZ_MIN_SPREAD / 2)
    min_coords[2] = min(min_coords[2], center_xz_coords[1] - XZ_MIN_SPREAD / 2)
    max_coords[2] = max(max_coords[2], center_xz_coords[1] + XZ_MIN_SPREAD / 2)

    fake_bbox_coordinates = np.random.uniform(min_coords, max_coords, size=(num_to_add, 7))
    bboxes_fake = [Bbox3d(coord, -1, seg_class_id=seg_class_id, confidence=99, instance_id=-1)
                   for coord in fake_bbox_coordinates]
    return bboxes + bboxes_fake


def drop_bboxes_from_clip(bboxes_per_frame: List[List[Bbox3d]], *,
                          bbox_drop_p: float) -> List[List[Bbox3d]]:
    if bbox_drop_p == 0:
        return bboxes_per_frame
    return [drop_bboxes(bboxes, bbox_drop_p=bbox_drop_p) for bboxes in bboxes_per_frame]


def jitter_bboxes_in_clip(bboxes_per_frame: List[List[Bbox3d]], *,
                          xz_std: float,  # 0.4
                          theta_std: float,  # 0.17 ~5 degrees
                          lwh_std: float,  # 0.3
                          ) -> List[List[Bbox3d]]:
    if xz_std == theta_std == lwh_std == 0:
        return bboxes_per_frame
    return [jitter_bboxes(bboxes, xz_std=xz_std, lwh_std=lwh_std, theta_std=theta_std)
            for bboxes in bboxes_per_frame]


def add_bboxes_to_clip(bboxes_per_frame: List[List[Bbox3d]], seg_class_id: int, *,
                       bbox_add_p: float, num_bboxes_to_always_add: int) -> List[List[Bbox3d]]:
    if bbox_add_p == 0 and num_bboxes_to_always_add == 0:
        return bboxes_per_frame
    return [add_bboxes(bboxes, seg_class_id,
                       bbox_add_p=bbox_add_p, num_bboxes_to_always_add=num_bboxes_to_always_add)
            for bboxes in bboxes_per_frame]
