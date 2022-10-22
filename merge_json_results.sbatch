#!/bin/bash
#SBATCH --job-name="merge_results"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --mail-type=END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT
#SBATCH --output=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
#SBATCH --error=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
cd ~/workspace/graphmot
srun python merge_json_results.py \
v1.0-trainval_tracking \
/storage/slurm/kimal/graphmot_workspace/nuscenes/val/centerpoint_mmdetection_cascade_nuimages \
1aggsections