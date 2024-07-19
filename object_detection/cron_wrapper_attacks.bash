#!/bin/bash -l

# Check if squeue command output contains only the header
if [ "$(squeue | wc -l)" -le 1 ]; then
    # If only the header is present, execute the Python script
    echo "squeue is empty, executing script"
    cd /pfs/work7/workspace/scratch/ma_jjakubas-team_project_fss2024/benchmarking_robustness/object_detection/
    /pfs/work7/workspace/scratch/ma_jjakubas-team_project_fss2024/miniconda3/envs/benchmark/bin/python submit_attack_tasks.py
    # /pfs/work7/workspace/scratch/ma_jjakubas-team_project_fss2024/miniconda3/envs/benchmark/bin/python submit_corruption_val_tasks.py
fiw