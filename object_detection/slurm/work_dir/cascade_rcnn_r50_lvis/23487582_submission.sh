#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --error=/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/slurm/work_dir/cascade_rcnn_r50_lvis/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=submitit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/slurm/work_dir/cascade_rcnn_r50_lvis/%j_0_log.out
#SBATCH --partition=gpu_4
#SBATCH --signal=USR2@90
#SBATCH --time=02:00:00
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/slurm/work_dir/cascade_rcnn_r50_lvis/%j_%t_log.out --error /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/slurm/work_dir/cascade_rcnn_r50_lvis/%j_%t_log.err /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/bin/python -u -m submitit.core._submit /pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/slurm/work_dir/cascade_rcnn_r50_lvis
