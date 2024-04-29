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

#snow
python attacks.py gma --pretrained_ckpt sintel --val_dataset sintel-val --attack weather --weather_steps 750  --weather_learn_offset True --weather_learn_motionoffset True --weather_learn_color True --weather_learn_transparency True --weather_alph_motion 1000. --weather_alph_motionoffset 1000. --weather_data /pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/DistractingDownpour/particles_3000_npz --weather_dataset Sintel --weather_dataset_stage training --attack_targeted True --weather_depth_check True
