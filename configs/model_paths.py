from dataset_classes.nuscenes.classes import NuScenesClasses
from dataset_classes.kitti.classes import KITTIClasses


offline_models = {
    NuScenesClasses.car:
    ("gnn_training",
     "21-09-12_08:07_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_car",
     "val_loss=0.002494-step=71039-epoch=1109"),

    NuScenesClasses.pedestrian:
    ("gnn_training",
     "21-09-12_14:31_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_pedestrian",
     "val_loss=0.014720-step=65519-epoch=1039"),

    NuScenesClasses.bicycle:
    ("gnn_training",
     "21-09-12_16:19_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_bicycle",
     "val_loss=0.018420-step=60799-epoch=949"),

    NuScenesClasses.bus:
    ("gnn_training",
     "21-09-12_14:40_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_bus",
     "val_loss=0.021843-step=64889-epoch=1029"),

    NuScenesClasses.motorcycle:
    ("gnn_training",
     "21-09-12_15:08_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_motorcycle",
     "val_loss=0.008153-step=44159-epoch=689"),

    NuScenesClasses.trailer:
    ("gnn_training",
     "21-09-12_15:01_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_trailer",
     "val_loss=0.005969-step=64259-epoch=1019"),

    NuScenesClasses.truck:
    ("gnn_training",
     "21-09-12_15:00_aug1_smaller_newaug_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_truck",
     "val_loss=0.015254-step=76799-epoch=1199"),
}


online_models = {
    KITTIClasses.car:
    ("gnn_training_online",
     "21-09-13_23:55_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-07_car",
     "val_loss=0.000151-step=87039-epoch=1359"),

    KITTIClasses.pedestrian:
    ("gnn_training_online",
     "21-09-13_23:57_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-12_pedestrian",
     "val_loss=0.001314-step=100169-epoch=1589"),



    NuScenesClasses.car:
    ("gnn_training_online",
     "21-09-13_23:55_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-07_car",
     "val_loss=0.000151-step=87039-epoch=1359"),

    NuScenesClasses.pedestrian:
    ("gnn_training_online",
     "21-09-13_23:57_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-12_pedestrian",
     "val_loss=0.001314-step=100169-epoch=1589"),

    NuScenesClasses.bicycle:
    ("gnn_training_online",
     "21-09-14_00:01_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-13_bicycle",
     "val_loss=0.001451-step=76549-epoch=1199"),

    NuScenesClasses.bus:
    ("gnn_training_online",
     "21-09-14_00:00_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-12_bus",
     "val_loss=0.001891-step=75599-epoch=1199"),

    NuScenesClasses.motorcycle:
    ("gnn_training_online",
     "21-09-14_00:01_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-13_motorcycle",
     "val_loss=0.000672-step=81919-epoch=1279"),

    NuScenesClasses.trailer:
    ("gnn_training_online",
     "21-09-14_00:01_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-12_trailer",
     "val_loss=0.004063-step=79379-epoch=1259"),

    NuScenesClasses.truck:
    ("gnn_training_online",
     "21-09-13_23:59_aug1_smaller_newaugonline_0.5xpos_max_sameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-12_truck",
     "val_loss=0.001291-step=106879-epoch=1669"),
}

#################
# Ablation models
#################

no_sameframe_models = {
    NuScenesClasses.car:
    ("gnn_training_ablation_done",
     "21-09-16_22:09_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_car",
     "val_loss=0.004384-step=71679-epoch=1119"),

    NuScenesClasses.pedestrian:
    ("gnn_training_ablation_done",
     "21-09-16_05:10_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_pedestrian",
     "val_loss=0.017228-step=74339-epoch=1179"),

    NuScenesClasses.bicycle:
    ("gnn_training_ablation_done",
     "21-09-16_05:10_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_bicycle",
     "val_loss=0.026433-step=44799-epoch=699"),

    NuScenesClasses.bus:
    ("gnn_training_ablation_done",
     "21-09-16_05:10_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_bus",
     "val_loss=0.039724-step=60479-epoch=959"),

    NuScenesClasses.motorcycle:
    ("gnn_training_ablation_done",
     "21-09-16_05:10_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_motorcycle",
     "val_loss=0.016370-step=39679-epoch=619"),

    NuScenesClasses.trailer:
    ("gnn_training_ablation_done",
     "21-09-16_05:10_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_trailer",
     "val_loss=0.012338-step=60479-epoch=959"),

    NuScenesClasses.truck:
    ("gnn_training_ablation_done",
     "21-09-16_05:10_aug1_smaller_newaug_offline_0.5xpos_max_nosameframe_recurr_edgedim16_steps4_focal_lr0.002_wd0.005_batch64_data21-09-06_truck",
     "val_loss=0.025497-step=55679-epoch=869"),
}
