from dataset_classes.nuscenes.classes import NuScenesClasses
from dataset_classes.kitti.classes import KITTIClasses


drop_nodes_p = {
    NuScenesClasses.car: 0.4,
    NuScenesClasses.pedestrian: 0.4,
    NuScenesClasses.bicycle: 0.6,
    NuScenesClasses.bus: 0.4,
    NuScenesClasses.motorcycle: 0.5,
    NuScenesClasses.trailer: 0.4,
    NuScenesClasses.truck: 0.4,
}

drop_edges_p = {
    NuScenesClasses.car: 0.2,
    NuScenesClasses.pedestrian: 0.2,
    NuScenesClasses.bicycle: 0.2,
    NuScenesClasses.bus: 0.2,
    NuScenesClasses.motorcycle: 0.2,
    NuScenesClasses.trailer: 0.2,
    NuScenesClasses.truck: 0.2,
}

dist_x_stds = {
    NuScenesClasses.car: 0.1,
    NuScenesClasses.pedestrian: 0.05,
    NuScenesClasses.bicycle: 0.1,
    NuScenesClasses.bus: 0.3,
    NuScenesClasses.motorcycle: 0.15,
    NuScenesClasses.trailer: 0.35,
    NuScenesClasses.truck: 0.3,
}

polar_z_stds = {  # "direction" between two detections
    NuScenesClasses.car: 0.1,
    NuScenesClasses.pedestrian: 0.1,
    NuScenesClasses.bicycle: 0.1,
    NuScenesClasses.bus: 0.2,
    NuScenesClasses.motorcycle: 0.15,
    NuScenesClasses.trailer: 0.25,
    NuScenesClasses.truck: 0.2,
}

theta_stds = {  # 99.7% are within 3x std
    # Maybe take 1/2 of the theta_std during graph construction
    NuScenesClasses.car: 0.1,
    NuScenesClasses.pedestrian: 0.3,  # might be too high? maybe take 0.2
    NuScenesClasses.bicycle: 0.25,
    NuScenesClasses.bus: 0.05,
    NuScenesClasses.motorcycle: 0.225,
    NuScenesClasses.trailer: 0.25,
    NuScenesClasses.truck: 0.1,
}