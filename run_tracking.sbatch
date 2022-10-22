#!/bin/bash
#SBATCH --job-name="mult0.4_track0.4_age3"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:4G
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT
#SBATCH --output=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
#SBATCH --error=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
cd ~/workspace/graphmot
srun python run_tracking.py nuscenes -max_past_dets 3 -initial_track_mult 0.4 -track_score_thres 0.4