#!/bin/bash
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:4
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --to_check_if_epochs_repeated_atss_r50_voc0712
#SBATCH --output=slurm/work_dir/to_check_if_epochs_repeated_atss_r50_voc0712.out


module load devel/cuda/11.8

bash mmdetection/tools/dist_train.sh \
    to_check_if_epochs_repeated_atss_r50_voc0712.py \
    4 \
    --work-dir slurm/work_dir/to_check_if_epochs_repeated_atss_r50_voc0712
