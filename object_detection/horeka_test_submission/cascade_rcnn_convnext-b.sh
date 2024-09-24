#!/bin/bash
#SBATCH --job-name=submission_cascade_rcnn_convnext-b # job name
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
    ./horeka_test_submission/cascade_rcnn_convnext-b_coco.py \
    --work-dir ./slurm/train_work_dir/cascade_rcnn_convnext-b_coco \
    --resume \
    --auto-scale-lr &

# GPU stats V2
while true; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv >> ./slurm/train_results/usage_cascade_rcnn_convnext-b.log
  sleep 60  # Adjust the interval as needed
done

# Wait for all background jobs to finish
wait