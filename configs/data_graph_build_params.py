from dataset_classes.nuscenes.classes import NuScenesClasses
from dataset_classes.kitti.classes import KITTIClasses


# At 50 km/h, will cover 1.39 m in a frame at 10Hz on KITTI
# 1.4 m/s - average human walking speed, 0.14 per frame
# NuScenes is at 2Hz
max_edge_distances = {  # in meters
    KITTIClasses.car: 3,  # should decrease or allow 2m noise?   # ideally 1.4
    KITTIClasses.pedestrian: 0.5,  # ideally 0.14
  
    NuScenesClasses.car: 10,          # 20.14 m, 18 time_diff on val; 15.6 on mini;   was 10
    NuScenesClasses.pedestrian: 1.2,  #  3.62 m, 14 time diff on val; 1.9 on mini;    was 1.5
    NuScenesClasses.bicycle: 5,       #  4.54 m,  7 time diff on val; 3.46 on mini;
    NuScenesClasses.bus: 10,          #  9.75 m,  4 time_diff on val; 5.57 on mini; 
    NuScenesClasses.motorcycle: 10,   # 10.25 m,  4 time_diff on val; 7.52 on mini; 
    NuScenesClasses.trailer: 6,       #  7 m,     4 time_diff on val; 0.115 on mini;
    NuScenesClasses.truck: 7,         # 10.43 m,  5 time_diff on val; 6.9 on mini;
}

xz_stds = {
    NuScenesClasses.car: 0.2,
    NuScenesClasses.pedestrian: 0.1,
    NuScenesClasses.bicycle: 0.15,
    NuScenesClasses.bus: 0.7,
    NuScenesClasses.motorcycle: 0.25,
    NuScenesClasses.trailer: 0.8,
    NuScenesClasses.truck: 0.7,
}

theta_stds = {  # 99.7% are within 3x std
    NuScenesClasses.car: 0.2,
    NuScenesClasses.pedestrian: 0.35,
    NuScenesClasses.bicycle: 0.5,
    NuScenesClasses.bus: 0.1,
    NuScenesClasses.motorcycle: 0.45,
    NuScenesClasses.trailer: 0.5,
    NuScenesClasses.truck: 0.2,
}

bbox_add_p = {  # 99.7% are within 3x std
    NuScenesClasses.car: 0.8,
    NuScenesClasses.pedestrian: 0.7,
    NuScenesClasses.bicycle: 0.9,
    NuScenesClasses.bus: 0.7,
    NuScenesClasses.motorcycle: 0.8,
    NuScenesClasses.trailer: 0.8,
    NuScenesClasses.truck: 0.7,
}

num_bboxes_to_always_add = {  # 99.7% are within 3x std
    NuScenesClasses.car: 3,
    NuScenesClasses.pedestrian: 3,
    NuScenesClasses.bicycle: 3,
    NuScenesClasses.bus: 1,
    NuScenesClasses.motorcycle: 3,
    NuScenesClasses.trailer: 1,
    NuScenesClasses.truck: 2,
}

