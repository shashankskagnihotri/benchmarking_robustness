#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --time=05:00:00
#SBATCH --job-name=Scenceflow_corruptions

echo Kitti_starts >> test 
cd /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/common_corruptions/
source ~/.bashrc
conda activate unified_env
python create_common_corruptions_kitti.py
