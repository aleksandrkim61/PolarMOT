#!/bin/bash
#SBATCH --job-name="car_no_delta"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:28G
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT
#SBATCH --output=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
#SBATCH --error=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
cd ~/workspace/graphmot
srun python train_nuscenes.py car --no_delta