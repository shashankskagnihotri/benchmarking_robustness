#!/bin/bash

# Parameters
#SBATCH --error=/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/log_test/%j/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/log_test/%j/%j_0_log.out
#SBATCH --partition=dev_single
#SBATCH --signal=USR2@90
#SBATCH --time=4
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/log_test/%j/%j_%t_log.out --error /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/log_test/%j/%j_%t_log.err /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/bin/python -u -m submitit.core._submit /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/log_test/%j
