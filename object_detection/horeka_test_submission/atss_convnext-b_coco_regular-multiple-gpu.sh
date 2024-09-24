#!/bin/bash
#SBATCH --job-name=atss_convnext-b_coco_regular-multiple-gpu # job name
#SBATCH --output=./slurm/train_results/log.%x.job_%j.out
#SBATCH --error=./slurm/train_results/log.%x.job_%j.err
#SBATCH --time=04:00:00         # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=gpu_4        # put the job into the gpu partition
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

## Nodes and tasks allocation
#SBATCH --nodes=2               # Request 2 nodes
#SBATCH --gres=gpu:1            # Request 1 GPU per node

## Load necessary modules
module load devel/cuda/11.8

# Run the training script with 2 nodes and 1 GPU per node
bash mmdetection/tools/dist_train.sh \
    ./horeka_test_submission/atss_convnext-b_coco_regular-multiple-gpu.py \
    1 \                             # Specify 1 GPU per node (since each node has only 1 GPU)
    --work-dir ./slurm/train_work_dir/atss_convnext-b_coco_regular-multiple-gpu \
    --resume \
    --auto-scale-lr \
    NNODES=2 &                      # Explicitly pass NNODES=2 to the dist_train.sh script

# Wait for all background jobs to finish
wait