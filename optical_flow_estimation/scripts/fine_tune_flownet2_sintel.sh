#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --job-name=SintelFlownet2
#SBATCH --output=slurm/Flownet2_Sintel.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/Flownet2Sintel_ERR.out

echo Flownet2Sintel

python ./mmflow/tools/train.py \
./custom_configs/flownet2_8x1_fine_sintel.py \
--load-from ./mmflow/checkpoints/flownet2_8x1_sfine_flyingthings3d_subset_384x768.pth 