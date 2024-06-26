#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --time=24:00:00
#SBATCH --job-name=Scenceflow_corruptions

cd /pfs/work7/workspace/scratch/ma_adackerm-team_project_fss2024/benchmarking_robustness/disparity_estimation/common_corruptions/
source ~/.bashrc
conda activate unified_env
python create_common_corruptions_sceneflow.py