#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=TestAttacks
#SBATCH --output=slurm/TestAttacks.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/TestAttacks.out

echo TestAttacks

python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack bim
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack bim --attack_targeted True
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack fgsm
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack fgsm --attack_targeted True
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack apgd
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack apgd --attack_targeted True
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack cospgd
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack cospgd --attack_targeted True
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack pgd
python attacks.py raft --pretrained_ckpt chairs --val_dataset chairs-val --attack pgd --attack_targeted True

