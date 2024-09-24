#!/bin/bash
#SBATCH --job-name=rtmdet_l_swin_b_p6_4xb16-100e_coco# job name
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
    ./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py \
    --work-dir ./slurm/train_work_dir/rtmdet_l_swin_b_p6_4xb16-100e_coco \
    --resume 

wait