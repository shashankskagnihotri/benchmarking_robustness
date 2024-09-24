#!/bin/bash
#SBATCH --job-name=atss_convnext-b_voc0712 # job name
#SBATCH --output=./slurm/train_results/log.%x.job_%j.out
#SBATCH --error=./slurm/train_results/log.%x.job_%j.err
#SBATCH --time=04:00:00         # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=gpu_4_a100 # put the job into the gpu partition
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

## Nodes and tasks allocation
#SBATCH --nodes=1               # number of nodes

## GPU allocation 
#SBATCH --gres=gpu:1            # number of GPUs per node

# Load necessary modules
module load devel/cuda/11.8

# Run each model training task on a separate GPU
python ./mmdetection/tools/train.py \
    ./horeka_test_submission/atss_convnext-b_voc0712.py \
    --work-dir ./slurm/train_work_dir/atss_convnext-b_voc0712 \
    --resume \
    --auto-scale-lr &


# Wait for all background jobs to finish
wait