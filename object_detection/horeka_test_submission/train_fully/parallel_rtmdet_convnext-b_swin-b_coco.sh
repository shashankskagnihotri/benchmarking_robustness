#!/bin/bash
#SBATCH --job-name=parallel_rtmdet_convnext-b_swin-b_coco # job name
#SBATCH --output=./slurm/train_work_dir/%x/$log.%x.job_%j.out
#SBATCH --error=./slurm/train_work_dir/%x/log.%x.job_%j.err
#SBATCH --time=15:00         # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=accelerated-h100 # put the job into the gpu partition
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

## Nodes and tasks allocation
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=2     # MPI processes per node (1 task per GPU)
#SBATCH --gpus-per-task=2 

## GPU allocation 
#SBATCH --gres=gpu:4            # number of GPUs per node

# Load necessary modules
module load devel/cuda/11.8

srun --gres=gpu:2 --ntasks=1 bash mmdetection/tools/dist_train.sh \
    ./horeka_test_submission/train_fully/rtmdet_convnext-b_coco.py \
    1 \
    --work-dir ./slurm/train_work_dir/$SLURM_JOB_ID/rtmdet_convnext-b_coco \
    --resume \
    --auto-scale-lr &

srun --gres=gpu:2 --ntasks=1 bash mmdetection/tools/dist_train.sh \
    ./horeka_test_submission/train_fully/rtmdet_swin-b_coco.py \
    3 \
    --work-dir ./slurm/train_work_dir/$SLURM_JOB_ID/rtmdet_swin-b_coco \
    --resume \
    --auto-scale-lr

# Wait for all background jobs to finish
wait