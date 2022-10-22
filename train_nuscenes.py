import time
from itertools import product
from typing import Dict, Union
import os
from pathlib import Path
import datetime
import traceback
import argparse

import numpy as np
import torch
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

from configs.local_variables import NUSCENES_WORK_DIR
from dataset_classes.nuscenes.classes import NuScenesClasses

from models.graph_tracker_offline import GraphTrackerOffline, _build_params_dict
from data.single_graph_dataset import SingleGraphDataset
from data.augmentation import DropNodes, DropEdges, JitterEdgeAttr
from configs.data_paths import (stored_train_offline_boston_graphs_folders,
                                stored_train_offline_singapore_graphs_folders,
                                stored_train_offline_graphs_folders,
                                stored_train_online_graphs_folders,
                                stored_train_offline_nodelta_graphs_folders,
                                stored_train_offline_cartesian_graphs_folders)
from configs.model_paths import offline_models
from configs.data_online_aug_params import drop_nodes_p, drop_edges_p, dist_x_stds, polar_z_stds, theta_stds

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("class_str", type=str, help="Class to run for, e.g. 'car', 'pedestrian', etc.")
parser.add_argument("-mpn_steps", type=int, default=4, help="Number of message passing steps")
parser.add_argument('--online', default=False, action='store_true')
parser.add_argument('--offline', dest="online", action='store_false')
parser.add_argument('--no_sameframe', default=False, action='store_true')
parser.add_argument('--continue_training', default=False, action='store_true')
parser.add_argument('--no_delta', default=False, action='store_true')
parser.add_argument('--cartesian', default=False, action='store_true')
parser.add_argument('--mini', default=False, action='store_true')
parser.add_argument("-edge_mlps_count", type=int, default=3, help="Number of distinct node MLPs")
parser.add_argument("-node_aggr_sections", type=int, default=3, help="Number of distinct sections in Node aggregation")
parser.add_argument("-area_filter", type=str, default="",
                    help="Area from which to take data, everything else will be ignored")
parser.add_argument('--no_val', default=False, action='store_true')
args = parser.parse_args()
seg_class_id = NuScenesClasses[args.class_str]
online = args.online
print(f"Training with {'on' if online else 'off'}line data for {seg_class_id}")
work_dir = NUSCENES_WORK_DIR

# Add a mapping for augmentation params for each class
drop_nodes_transform = DropNodes(drop_nodes_p[seg_class_id])
drop_edges_transform = DropEdges(drop_edges_p[seg_class_id] * 2 if online else drop_edges_p[seg_class_id],
                                 online_past_edges=online)  # drop fewer edges in offline mode ?
jitter_transform = JitterEdgeAttr(dist_x_std=dist_x_stds[seg_class_id],
                                  polar_z_std=polar_z_stds[seg_class_id],
                                  theta_std=theta_stds[seg_class_id])
transforms_all = Compose([drop_nodes_transform, drop_edges_transform, jitter_transform])

# # To load
if args.no_delta:
    folder_to_load = stored_train_offline_nodelta_graphs_folders[seg_class_id]
elif args.cartesian:
    folder_to_load = stored_train_offline_cartesian_graphs_folders[seg_class_id]
elif args.area_filter == "singapore":
    folder_to_load = stored_train_offline_singapore_graphs_folders[seg_class_id]
elif args.area_filter == "boston":
    folder_to_load = stored_train_offline_boston_graphs_folders[seg_class_id]
else:
    if online:
        folder_to_load = stored_train_online_graphs_folders[seg_class_id]
    else:
        folder_to_load = stored_train_offline_graphs_folders[seg_class_id]

data_path = Path(work_dir) / "trainval" / folder_to_load
data_train_list = SingleGraphDataset(data_path, f"{'mini_' if args.mini else ''}train", transforms_all)
print("Loaded train set")

if args.no_val:
    data_val_list = SingleGraphDataset(data_path, f"{'mini_' if args.mini else ''}train", transforms_all)
else:
    data_val_list = SingleGraphDataset(data_path, "val", transforms_all)
print("Loaded val set")

NUM_WORKERS = 2  # 4
# 32 fits on NuScenes car clips=11
# 64 is good on KITTI Car. 128 for Ped on KITTI
# 128 for online KITTI
# 128 Nu regular size >38G, larger size >42G
# 64 Nu smaller size >17G
batch_size = 64

log_name = f"gnn_training_ablation{'_online' if online else ''}"
# log_name = "gnn_training_slurm"
# log_name = "gnn_training_mini_slurm"
if args.area_filter:
    log_name += f"_{args.area_filter}"
if args.edge_mlps_count != 3:
    log_name += f"_{args.edge_mlps_count}nodeMLPs"
if args.node_aggr_sections != 3:
    log_name += f"_{args.node_aggr_sections}nodeAggr"
log_folder_base_name = Path(work_dir)

# base_exp_name = "aug2_0.5xpos_slope0.2_world_polar_pretrained"
# base_exp_name = "aug2_0.5xpos_world_polar_train_mini_val_full"
# base_exp_name = "aug2_0.5xpos_world_no_polardelta_train_mini_val_full"
# base_exp_name = "aug2_0.5xpos_notimeaware_less_params_fixed_graph"
# base_exp_name = "aug2_0.5xpos_world_polar_bs64_more_params"
# base_exp_name = "aug2_0.5xpos_world_polar_fewer_params_full_val"
# base_exp_name = "aug2_0.5xpos_world_polar_bs64_full_len"
# base_exp_name = "aug1_0.5xpos_world_polar_full_len_fullval_smaller"
base_exp_name = f"aug1_newaug_{'on' if online else 'off'}line"

nonlinearity_common = nn.LeakyReLU(inplace=True, negative_slope=0.2)  # nn.Tanh()
use_batchnorm_list = [False]
# Initial edge model
initial_edge_model_input_dim_list = [4]  # fixed
edge_dim_list = [16]  # [16, 32]
fc_dims_initial_edge_model_multipliers_list = [(1, 1)]  # (1, 1)
nonlinearity_initial_edge_list = [nonlinearity_common]

# Initial node model
fc_dims_initial_node_model_multipliers_list = [(2, 4, 1)]  # (1, 2, 1), (2,4,1)
nonlinearity_initial_node_list = [nonlinearity_common]

# Edge model
fc_dims_edge_model_multipliers_list = [(4, 1)]  # (6,4,1), (4,1)
nonlinearity_edge_list = [nonlinearity_common]

# TimeAware Node model
fc_dims_directed_flow_model_multipliers_list = [(2, 1)]  # (4,2,1), (2,1); (4,2,2,1) for online
nonlinearity_directed_flow_list = [nonlinearity_common]
node_model_agg_list = ["max"]  # ["max", "attention", "attention_classifier", "attention_normalized"]
# multiplies node_dim, last layer output is always 1 [None, (2,1)]
fc_dims_node_attention_model_multipliers_list = [None]

# (4,2,1), (6,4,2,1)  (2,1) online - just an extra MLP
fc_dims_total_flow_model_multipliers_list = [(4, 2, 1)]
nonlinearity_total_flow_list = [nonlinearity_common]

# Edge classification model
fc_dims_edge_classification_model_multipliers_list = [
    (4, 2, 1,)]  # (2,1) [(0.5, ), None]  # mutliplies edge_dim
nonlinearity_edge_classification_list = [nonlinearity_common]

mpn_steps_list = [args.mpn_steps]
is_recurrent_list = [True]
node_dim_multiplier_list = [2]
pos_weight_multiplier_list = [0.5]  # 0.5

use_timeaware_list = [True]
use_same_frame_list = [not args.no_sameframe]
use_separate_edge_model_list = [False]
use_initial_node_model_list = [True]
edge_mlps_count_list = [args.edge_mlps_count]
node_aggr_sections_list = [args.node_aggr_sections]

lr_list = [2e-3]  # 2e-3 for batch=32 on both datasets
wd_list = [0.005]  # 0.15 for Nu, 0.3 on mini, 0.005 KITTI. 0.02
loss_type_list = ["focal"]  # ["bce", "focal"]
optimizer_list = ["radam"]  # ["radam"]  radam seems more unstable

scheduler_params_list = [
    # {"T_0": 40, "T_mult":1, "eta_min": 1e-4},  # on train val full with 0.15 limits
    {"T_0": 80, "T_mult": 1, "eta_min": 5e-5},  # on full trainval, more hops between local minimals
]

seeds = [123]  # 123, 456, 789, 0
_trainer_params: Dict[str, Union[str, float, int]] = {
    "max_epochs": 1200, "limit_train_batches": 0.15, "limit_val_batches": 0.15}
if args.mini:
    _trainer_params["limit_train_batches"] = 1.0

if args.continue_training:
    if seg_class_id == NuScenesClasses.car:
        if args.mpn_steps == 3:
            _trainer_params = {"max_epochs": 400, "limit_train_batches": 0.15, "limit_val_batches": 0.15,
                               "pretrained_runs_folder": "gnn_training_ablation_done",
                               "pretrained_folder": "21-09-16_17:05_aug1_smaller_newaug_offline_0.5xpos_max_sameframe_recurr_edgedim16_steps3_focal_lr0.002_wd0.005_batch64_data21-09-06_car",
                               "pretrained_ckpt": "val_loss=0.003121-step=51199-epoch=799",
                               }
elif online:
    _trainer_params = {"max_epochs": 600, "limit_train_batches": 0.15, "limit_val_batches": 0.15,
                       "pretrained_runs_folder": offline_models[seg_class_id][0],
                       "pretrained_folder": offline_models[seg_class_id][1],
                       "pretrained_ckpt": offline_models[seg_class_id][2],
                       }
trainer_params_list = [_trainer_params]

param_combos = list(product(initial_edge_model_input_dim_list,
                            edge_dim_list, fc_dims_initial_edge_model_multipliers_list, nonlinearity_initial_edge_list,
                            fc_dims_initial_node_model_multipliers_list, nonlinearity_initial_node_list,
                            node_model_agg_list, fc_dims_node_attention_model_multipliers_list,
                            fc_dims_edge_model_multipliers_list, nonlinearity_edge_list,
                            fc_dims_directed_flow_model_multipliers_list, nonlinearity_directed_flow_list,
                            fc_dims_total_flow_model_multipliers_list, nonlinearity_total_flow_list,
                            fc_dims_edge_classification_model_multipliers_list, nonlinearity_edge_classification_list,
                            use_batchnorm_list,
                            mpn_steps_list, is_recurrent_list, node_dim_multiplier_list, pos_weight_multiplier_list,
                            use_timeaware_list, use_same_frame_list, use_separate_edge_model_list, use_initial_node_model_list,
                            edge_mlps_count_list,
                            node_aggr_sections_list,
                            lr_list, wd_list,
                            loss_type_list,
                            seeds,
                            optimizer_list,
                            scheduler_params_list,
                            trainer_params_list,
                            ))
print(GraphTrackerOffline(_build_params_dict(*param_combos[0])))

data_train_list.shuffle()
data_val_list.shuffle()
dataloader_train = DataLoader(data_train_list, shuffle=True, drop_last=True,
                              batch_size=batch_size, num_workers=NUM_WORKERS)
dataloader_val = DataLoader(data_val_list, shuffle=False, drop_last=False,
                            batch_size=batch_size, num_workers=NUM_WORKERS)

print("Train batches", len(dataloader_train))
print("Val   batches", len(dataloader_val))

log_folder_name = log_folder_base_name / log_name
print(f"Running {len(param_combos)} combinations")
for param_i, params in enumerate(param_combos):
    params_dict = _build_params_dict(*params)
    trainer_params = params_dict["trainer_params"]
    for k, v in params_dict.items():
        os.write(1, f"{k:>45}: {v}\n".encode())

    seed = params_dict["seed"]
    seed_everything(seed)  # 456

    graph_tracker = GraphTrackerOffline(params_dict)

    edge_dim = params_dict["edge_dim"]
    mpn_steps = params_dict['mpn_steps']
    is_recurrent = params_dict["is_recurrent"]
    use_bn = params_dict['use_batchnorm']
    use_same_frame = params_dict['use_same_frame']
    edge_mlps_count = params_dict['edge_mlps_count']
    node_aggr_sections = params_dict['node_aggr_sections']

    # only makes sense when using intraframe
    use_separate_edge_model = use_same_frame and params_dict['use_separate_edge_model']
    use_initial_node_model = params_dict['use_initial_node_model']
    loss_type = params_dict['loss_type']
    lr = params_dict['lr']
    wd = params_dict['wd']

    log_name = f"{datetime.datetime.now().strftime('%y-%m-%d_%H:%M')}_{base_exp_name}_{params_dict['pos_weight_multiplier']}xpos"
    log_name += f"_{params_dict['directed_flow_agg']}"
    log_name += f"_{'' if use_same_frame else 'no'}sameframe{'_separate_edges' if use_separate_edge_model else ''}"
    log_name += f"_{edge_mlps_count}nodeMLPs"
    log_name += f"_{node_aggr_sections}nodeAggr"
    log_name += f"_{'' if is_recurrent else 'non'}recurr{'_zeronodes' if not use_initial_node_model else ''}"
    log_name += f"_edgedim{edge_dim}_steps{mpn_steps}{'_bn' if use_bn else ''}_{loss_type}_lr{lr}_wd{wd}_batch{batch_size}"
    log_name += f"_{'' if not args.no_delta else 'no'}delta{'polar' if not args.cartesian else 'cartesian'}"
    log_name += f"{'_mini' if args.mini else ''}_data{folder_to_load[:8]}_{seg_class_id.name}"
    os.write(1, f"{log_name}\n\n\n".encode())

    tensorboard_logger = TensorBoardLogger(str(log_folder_name), name=log_name)
    tensorboard_logger.log_hyperparams(params_dict)  # for the board and when continuing after pretraining

    # always have tensorboard logger first - to log histograms
    loggers = [tensorboard_logger]  # wandb_logger

    save_best_val_loss_checkpoint = ModelCheckpoint(dirpath=Path(tensorboard_logger.log_dir) / "checkpoints",
                                                    filename="{val_loss:.6f}-{step}-{epoch}",
                                                    monitor="val_loss",
                                                    mode="min",
                                                    save_top_k=-1,  # 20
                                                    save_last=True,
                                                    every_n_val_epochs=10,
                                                    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # maybe only do swa after full training and run it with SGD, not RAdam
    # swa = StochasticWeightAveraging(swa_epoch_start=0.9, swa_lrs=lr)
    callbacks = [save_best_val_loss_checkpoint, lr_monitor]  # swa

    if "pretrained_folder" in trainer_params:
        _log_folder_pretrained_name = Path(work_dir) / trainer_params["pretrained_runs_folder"]
        _folder_name = trainer_params["pretrained_folder"]
        _ckpt_name = trainer_params["pretrained_ckpt"]
        _ckpt_epoch = int(_ckpt_name.split("epoch=")[-1])
        _ckpt_path = str(_log_folder_pretrained_name /
                         f"{_folder_name}/version_0/checkpoints/{_ckpt_name}.ckpt")
        hparams_file = _ckpt_path.split("checkpoints")[0] + "hparams.yaml"
        graph_tracker = GraphTrackerOffline.load_from_checkpoint(_ckpt_path, hparams_file=hparams_file)
    else:
        print("Not loading a ckpt")
        _ckpt_epoch = 0
        _ckpt_path = None

    try:
        trainer = pl.Trainer(max_epochs=_ckpt_epoch + trainer_params["max_epochs"], gpus=1,
                             callbacks=callbacks, logger=loggers,
                             check_val_every_n_epoch=2,  # 1, 5
                             limit_train_batches=float(trainer_params["limit_train_batches"]),
                             limit_val_batches=float(trainer_params["limit_val_batches"]),
                             track_grad_norm=2,
                             terminate_on_nan=True,
                             gradient_clip_val=1,
                             gradient_clip_algorithm="norm",  # "norm" by default, can be "value"
                             resume_from_checkpoint=_ckpt_path,
                             )

        start_time = time.time()
        print(f"Starting {log_name} for {_ckpt_epoch + trainer_params['max_epochs']}")
        trainer.fit(graph_tracker, dataloader_train, dataloader_val)
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error for {log_folder_name}/{log_name}\n{e}\n{trace}")
        os.write(1, f"{str(trace)}\n".encode())

    finally:
        duration_m = (time.time() - start_time) / 60
        print(
            f"Training {log_name} for {int(trainer.logged_metrics['epoch'].item())} epochs took {duration_m:.2f} minutes")

    os.write(
        1, f"Finished training {param_i+1}/{len(param_combos)} param combo {datetime.datetime.now().strftime('%y-%m-%d_%H:%M')}\n".encode())

os.write(1, f"Finished all training {datetime.datetime.now().strftime('%y-%m-%d_%H:%M')}\n".encode())
