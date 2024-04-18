#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --job-name=TestAttacks
#SBATCH --output=slurm/TestAttacks.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/TestAttacks.out

echo TestAttacks

python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack bim
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack bim --attack_targeted True
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack fgsm
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack fgsm --attack_targeted True
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack apgd
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack apgd --attack_targeted True
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack cospgd
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack cospgd --attack_targeted True
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack pgd
python attacks.py raft --pretrained_ckpt things --val_dataset kitti-2012+kitti-2015 --attack pgd --attack_targeted True


