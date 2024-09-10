#!/bin/bash
#SBATCH --job-name=test_submission_atss_and_cascade_rcnn_for_convnext_and_swin-b  # job name
#SBATCH --output=../slurm/train_results/log.%x.job_%j.out  # corrected path
#SBATCH --error=../slurm/train_results/log.%x.job_%j.err   # corrected path
#SBATCH --time=00:10:00         # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=accelerated # put the job into the gpu partition
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

## Nodes and tasks allocation
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=4     # MPI processes per node (1 task per GPU)

## GPU allocation 
#SBATCH --gres=gpu:4            # number of GPUs per node

# Load necessary modules
# module load devel/cuda/11.8
module load devel/cuda/12.2

# Run each model training task on a separate GPU
srun --gres=gpu:1 --ntasks=1 --exclusive python ../mmdetection/tools/train.py \
    ../horeka_test_submission/atss_convnext-b_coco.py \
    --work-dir ../slurm/train_work_dir/atss_convnext-b_coco \
    --resume \
    --auto-scale-lr &

srun --gres=gpu:1 --ntasks=1 --exclusive python ../mmdetection/tools/train.py \
    ../horeka_test_submission/atss_swin-b_coco.py \
    --work-dir ../slurm/train_work_dir/atss_swin-b_coco \
    --resume \
    --auto-scale-lr &

srun --gres=gpu:1 --ntasks=1 --exclusive python ../mmdetection/tools/train.py \
    ../horeka_test_submission/cascade_rcnn_convnext-b_coco.py \
    --work-dir ../slurm/train_work_dir/cascade_rcnn_convnext-b_coco \
    --resume \
    --auto-scale-lr &

srun --gres=gpu:1 --ntasks=1 --exclusive python ../mmdetection/tools/train.py \
    ../horeka_test_submission/cascade_rcnn_swin-b_coco.py \
    --work-dir ../slurm/train_work_dir/cascade_rcnn_swin-b_coco \
    --resume \
    --auto-scale-lr &

# GPU stats V1
# Check GPU utilization during the job
nvidia-smi

# Wait for all background jobs to finish
wait