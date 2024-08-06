#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de
#SBATCH -o ./logs/output.%a.out # STDOUT

# Config
# CHECKPOINT_FILE="mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
# CONFIG_FILE="mmdetection/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"
# CONFIG_FILE="models/DINO_Swin-L/DINO_Swin-L.py"
# CHECKPOINT_FILE="models/DINO_Swin-L/latest.pth"
# CONFIG_FILE="models/TOOD_R-101-dcnv2/TOOD_R-101-dcnv2.py"
# CHECKPOINT_FILE="models/TOOD_R-101-dcnv2/latest.pth"
CONFIG_FILE="models/VarifocalNet_R-101-FPN/VarifocalNet_R-101-FPN.py"
CHECKPOINT_FILE="models/VarifocalNet_R-101-FPN/latest.pth"

ALPHA=2.55
STEPS=2
EPSILON=8
ATTACK="pgd"  # "pgd", "fgsm", "cospgd", "none"

cd .. # expects object_detection/slurm to be the working directory

echo "#########  START ATTACK ###########"
echo "CHECKPOINT_FILE="${CHECKPOINT_FILE}
echo "CONFIG_FILE="${CONFIG_FILE}
echo "ALPHA="${ALPHA}
echo "STEPS="${STEPS}
echo "EPSILON="${EPSILON}
echo "ATTACK="${ATTACK}

python -m pudb adv_attack.py --config_file ${CONFIG_FILE} --checkpoint_file ${CHECKPOINT_FILE} --steps ${STEPS} --alpha ${ALPHA} --epsilon ${EPSILON} --attack ${ATTACK} 