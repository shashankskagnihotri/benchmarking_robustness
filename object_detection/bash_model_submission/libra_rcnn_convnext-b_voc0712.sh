#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=libra_rcnn_convnext-b_voc0712_Job
#SBATCH --output=/pfs/work7/workspace/scratch/ma_skral-team_project_fss2024/slurm/libra_rcnn_convnext-b_voc0712.out

module load devel/cuda/11.8

python tools/train.py \
    ./configs_to_train/libra_rcnn_convnext-b_voc0712.py \
    --work-dir ./bash_model_submission/output/libra_rcnn_convnext-b_voc0712 \
    --resume \
    --auto-scale-lr