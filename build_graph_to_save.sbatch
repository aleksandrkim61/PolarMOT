#!/bin/bash
#SBATCH --job-name="car_no_delta_graph"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=150G
#SBATCH --time=05:50:00
#SBATCH --mail-type=END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT
#SBATCH --output=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
#SBATCH --error=/storage/slurm/kimal/graphmot_workspace/nuscenes/trainval/slurm/slurm-%j.out
cd ~/workspace/graphmot
srun python build_graph_to_save.py car --no_val --no_delta