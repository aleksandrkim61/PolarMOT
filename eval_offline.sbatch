#!/bin/bash
#SBATCH --job-name="no_delta_car"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:4G
#SBATCH --mem=30G
#SBATCH --time=01:20:00
#SBATCH --mail-type=END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT
#SBATCH --output=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
#SBATCH --error=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
cd ~/workspace/graphmot
srun python eval_offline.py nuscenes car --no_delta