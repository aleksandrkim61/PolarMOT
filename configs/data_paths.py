from dataset_classes.nuscenes.classes import NuScenesClasses
from dataset_classes.kitti.classes import KITTIClasses


# augmentations as transforms
det_graphs_val_det0 = {
    NuScenesClasses.car:        "21-09-01_13:56_dets_offline_deltapolar_fulllen_sameframe_car",
    NuScenesClasses.pedestrian: "21-09-12_dets_offline_deltapolar_fulllen_sameframe_pedestrian",
    NuScenesClasses.bicycle:    "21-09-12_dets_offline_deltapolar_fulllen_sameframe_bicycle",
    NuScenesClasses.bus:        "21-09-12_dets_offline_deltapolar_fulllen_sameframe_bus",
    NuScenesClasses.motorcycle: "21-09-12_dets_offline_deltapolar_fulllen_sameframe_motorcycle",
    NuScenesClasses.trailer:    "21-09-12_dets_offline_deltapolar_fulllen_sameframe_trailer",
    NuScenesClasses.truck:      "21-09-12_dets_offline_deltapolar_fulllen_sameframe_truck",
}
det_graphs_val_det02 = {
    NuScenesClasses.car:        "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_car",
    NuScenesClasses.pedestrian: "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_pedestrian",
    NuScenesClasses.bicycle:    "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_bicycle",
    NuScenesClasses.bus:        "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_bus",
    NuScenesClasses.motorcycle: "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_motorcycle",
    NuScenesClasses.trailer:    "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_trailer",
    NuScenesClasses.truck:      "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_truck",
}
det_graphs_val_det03 = {
    # NuScenesClasses.car:        "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_car",
    NuScenesClasses.pedestrian: "21-09-15_dets_offline_deltapolar_fulllen_sameframe_det0.3_pedestrian",
    NuScenesClasses.bicycle:    "21-09-15_dets_offline_deltapolar_fulllen_sameframe_det0.3_bicycle",
    # NuScenesClasses.bus:        "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_bus",
    NuScenesClasses.motorcycle: "21-09-15_dets_offline_deltapolar_fulllen_sameframe_det0.3_motorcycle",
    # NuScenesClasses.trailer:    "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_trailer",
    # NuScenesClasses.truck:      "21-09-13_dets_offline_deltapolar_fulllen_sameframe_det0.2_truck",
}
det_graphs_mini_val = {
    NuScenesClasses.car: "21-09-05_23:46_dets_offline_deltapolar_fulllen_sameframe_car",
}

# ### NuScenes trained on mini with polar/delta
# ckpt_root_folder = "gnn_training_mini"
# ckpt_folder_name = "21-08-27_01:52_aug2_0.5xpos_world_polar_train_mini_val_full_offline__recurr_edgedim16_steps4_focal_lr0.002_wd0.3_clip11_batch32_car"
# ckpt_name = "val_loss=0.010081-step=7789-epoch=409"
# ckpt_folder_name = "21-08-27_20:16_aug2_0.5xpos_world_polar_train_mini_val_full_offline__recurr_edgedim16_steps4_focal_lr0.002_wd0.15_clip11_batch32_car"
# ckpt_name = "val_loss=0.020341-step=6649-epoch=349"
# ckpt_folder_name = "21-08-30_00:11_aug2_0.5xpos_world_polar_full_len_offline__recurr_edgedim16_steps4_focal_lr0.001_wd0.15_clip11_batch16_car"  # polar
# ckpt_name = "val_loss=0.004825-step=14039-epoch=359"

# # ### KITTI world
# ckpt_root_folder = "gnn_training"
# ckpt_folder_name = "21-08-21_04:18_aug2_0.5xpos_slope0.2_world_polar_offline_recurr_edgedim16_steps4_focal_lr0.004_wd0.005_clip11_batch64_car"
# ckpt_name = "val_loss=0.016502-epoch=51"


stored_train_offline_graphs_folders = {
    NuScenesClasses.car:        "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_car",
    NuScenesClasses.pedestrian: "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_pedestrian",
    NuScenesClasses.bicycle:    "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_bicycle",
    NuScenesClasses.bus:        "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_bus",
    NuScenesClasses.motorcycle: "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_motorcycle",
    NuScenesClasses.trailer:    "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_trailer",
    NuScenesClasses.truck:      "21-09-06_aug1_deltapolar_full_len_sameframe_nodrop_truck",
}

stored_train_offline_boston_graphs_folders = {
    NuScenesClasses.car:        "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_car",
    NuScenesClasses.pedestrian: "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_pedestrian",
    NuScenesClasses.bicycle:    "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_bicycle",
    NuScenesClasses.bus:        "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_bus",
    NuScenesClasses.motorcycle: "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_motorcycle",
    NuScenesClasses.trailer:    "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_trailer",
    NuScenesClasses.truck:      "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_boston_truck",
}

stored_train_offline_singapore_graphs_folders = {
    NuScenesClasses.car:        "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_car",
    NuScenesClasses.pedestrian: "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_pedestrian",
    NuScenesClasses.bicycle:    "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_bicycle",
    NuScenesClasses.bus:        "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_bus",
    NuScenesClasses.motorcycle: "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_motorcycle",
    NuScenesClasses.trailer:    "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_trailer",
    NuScenesClasses.truck:      "22-02-24_aug1_lenfull_deltapolar_sameframe_offline_11_nodrop_singapore_truck",
}

stored_train_online_graphs_folders = {
    NuScenesClasses.car:        "21-09-07_aug1_deltapolar_full_len_sameframe_online_20_nodrop_car",
    NuScenesClasses.pedestrian: "21-09-12_aug1_deltapolar_full_len_sameframe_online_20_nodrop_pedestrian",
    NuScenesClasses.bicycle:    "21-09-13_aug1_deltapolar_full_len_sameframe_online_20_nodrop_bicycle",
    NuScenesClasses.bus:        "21-09-12_aug1_deltapolar_full_len_sameframe_online_20_nodrop_bus",
    NuScenesClasses.motorcycle: "21-09-13_aug1_deltapolar_full_len_sameframe_online_20_nodrop_motorcycle",
    NuScenesClasses.trailer:    "21-09-12_aug1_deltapolar_full_len_sameframe_online_20_nodrop_trailer",
    NuScenesClasses.truck:      "21-09-12_aug1_deltapolar_full_len_sameframe_online_20_nodrop_truck",
}

stored_train_offline_nodelta_graphs_folders = {
    NuScenesClasses.car:        "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_car",
    NuScenesClasses.pedestrian: "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_pedestrian",
    NuScenesClasses.bicycle:    "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_bicycle",
    NuScenesClasses.bus:        "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_bus",
    NuScenesClasses.motorcycle: "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_motorcycle",
    NuScenesClasses.trailer:    "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_trailer",
    NuScenesClasses.truck:      "21-09-17_aug1_nodeltapolar_lenfull_sameframe_offline_20_nodrop_truck",
}

stored_train_offline_cartesian_graphs_folders = {
    NuScenesClasses.car:        "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_car",
    NuScenesClasses.pedestrian: "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_pedestrian",
    NuScenesClasses.bicycle:    "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_bicycle",
    NuScenesClasses.bus:        "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_bus",
    NuScenesClasses.motorcycle: "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_motorcycle",
    NuScenesClasses.trailer:    "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_trailer",
    NuScenesClasses.truck:      "21-09-17_aug1_deltacartesian_lenfull_sameframe_offline_20_nodrop_truck",
}
