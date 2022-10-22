MOUNT_PATH = ""  # in case you are mounting data storage externally
SPLIT = "val"  # "training" / "testing" for KITTI, "val" / "test" for NuScenes - actual splits have to be passed as args

KITTI_WORK_DIR = MOUNT_PATH + "/storage/slurm/kimal/graphmot_workspace/kitti"
KITTI_DATA_DIR = MOUNT_PATH + "/storage/slurm/osep/datasets/kitti"

NUSCENES_WORK_DIR = MOUNT_PATH + "/storage/slurm/kimal/graphmot_workspace/nuscenes"
# NUSCENES_DATA_DIR = MOUNT_PATH + "/storage/slurm/kimal/datasets_original/nuscenes"
NUSCENES_DATA_DIR = MOUNT_PATH + "/storage/slurm/shil/nuscenes/trainval"
