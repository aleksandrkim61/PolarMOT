from dataset_classes.nuscenes.classes import NuScenesClasses
from dataset_classes.kitti.classes import KITTIClasses

# "mot_det_0_0_0_0_0_0_0_det0_mean_car_21-09-12_08:07_3d_best",
# "mot_det_0.2_0.2_0.2_0_det0.3_mean_pedestrian_21-09-12_14:31_3d_best",
# "mot_det_0_0_0_0_0_0_0_det0.0_mean_bicycle_21-09-12_16:19_3d_best",
# "mot_det_0_0_0_0_0_0_0_det0_mean_bus_21-09-12_14:40_3d_best",
# "mot_det_0_0_0_0_0_0_0_det0.2_mean_motorcycle_21-09-12_15:08_3d_best",
# "mot_det_0_0_0_0_0_0_0_det0_mean_trailer_21-09-12_15:01_3d_best",
# "mot_det_0_0_0_0_0_0_0_det0_mean_truck_21-09-12_15:00_3d_best"

det_thresholds_offline = {
    KITTIClasses.car: 0,
    KITTIClasses.pedestrian: 0,

    NuScenesClasses.car: 0,
    NuScenesClasses.pedestrian: 0.3,
    NuScenesClasses.bicycle: 0,
    NuScenesClasses.bus: 0,
    NuScenesClasses.motorcycle: 0.2,
    NuScenesClasses.trailer: 0,
    NuScenesClasses.truck: 0,
}

track_pred_thresholds_offline = {
    KITTIClasses.car: 0.3,  # 0.4 could be better
    KITTIClasses.pedestrian: 0.2,

    NuScenesClasses.car: 0.7,
    NuScenesClasses.pedestrian: 0.5,
    NuScenesClasses.bicycle: 0.8,
    NuScenesClasses.bus: 0.7,
    NuScenesClasses.motorcycle: 0.7,
    NuScenesClasses.trailer: 0.7,
    NuScenesClasses.truck: 0.7,
}

track_pred_thresholds_online = {
    KITTIClasses.car: 0.3,  # 0.4 could be better
    KITTIClasses.pedestrian: 0.2,

    NuScenesClasses.car: 0.3,  # 0.4 could be better
    NuScenesClasses.pedestrian: 0.2,
    NuScenesClasses.bicycle: 0.3,
    NuScenesClasses.bus: 0.1,  # 0.2 could be better
    NuScenesClasses.motorcycle: 0.2,
    NuScenesClasses.trailer: 0.3,
    NuScenesClasses.truck: 0.3,
}

track_initial_mult_online = {
    KITTIClasses.car: 0.3,  # 0.4 could be better
    KITTIClasses.pedestrian: 0.2,

    NuScenesClasses.car: 0.3,  # 0.4 could be better
    NuScenesClasses.pedestrian: 0.2,
    NuScenesClasses.bicycle: 0.3,
    NuScenesClasses.bus: 0.1,  # 0.2 could be better
    NuScenesClasses.motorcycle: 0.2,
    NuScenesClasses.trailer: 0.3,
    NuScenesClasses.truck: 0.3,
}
"""
Best so far with max_age3:
         det_score     pred_score,min_mult
"bicycle":     0.0 and 0.3_0.3
"bus":         0.0 and 0.1_0.1
"car":         0.0 and 0.3_0.3
"motorcycle":  0.0 and 0.2_0.2
"pedestrian":  0.2 and 0.2_0.2
"trailer":     0.2 and 0.3_0.3
"truck":       0.0 and 0.3_0.3

"bicycle": 0.5385903231791607
"bus": 0.8065759966999139
"car": 0.7990385813031249
"motorcycle": 0.6301970323262932
"pedestrian": 0.7768163993434639
"trailer": 0.47194940265295465
"truck": 0.6425941313340726

average AMOTA: 0.66653741
"""

max_online_ages = {
    KITTIClasses.car: 3,
    KITTIClasses.pedestrian: 3,

    NuScenesClasses.car: 3,
    NuScenesClasses.pedestrian: 3,
    NuScenesClasses.bicycle: 3,
    NuScenesClasses.bus: 3,
    NuScenesClasses.motorcycle: 3,
    NuScenesClasses.trailer: 3,
    NuScenesClasses.truck: 3,
}
