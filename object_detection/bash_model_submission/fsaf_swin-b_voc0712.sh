#!/bin/bash
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fsaf_swin-b_voc0712_Job
#SBATCH --output=/pfs/work7/workspace/scratch/ma_skral-team_project_fss2024/slurm/fsaf_swin-b_voc0712.out

module load devel/cuda/11.8

python tools/train.py \
    ./configs_to_train/fsaf_swin-b_voc0712.py \
    --work-dir ./bash_model_submission/output/fsaf_swin-b_voc0712 \
    --resume \
    
    