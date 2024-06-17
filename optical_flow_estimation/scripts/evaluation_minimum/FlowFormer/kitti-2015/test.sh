#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --job-name=test
#SBATCH --output=slurm/test.out
#SBATCH --error=slurm/test_err.out

cd ../../../../
python attacks.py flowformer --pretrained_ckpt kitti --val_dataset kitti-2015