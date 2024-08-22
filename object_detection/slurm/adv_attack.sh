#!/bin/bash
#SBATCH --partition=gpu_4,gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de
#SBATCH -o ./logs/output.%j.out # STDOUT

# Config
# CHECKPOINT_FILE="mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
# CONFIG_FILE="mmdetection/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"
# CONFIG_FILE="models/DINO_Swin-L/DINO_Swin-L.py"
# CHECKPOINT_FILE="models/DINO_Swin-L/latest.pth"
# CONFIG_FILE="models/TOOD_R-101-dcnv2/TOOD_R-101-dcnv2.py"
# CHECKPOINT_FILE="models/TOOD_R-101-dcnv2/latest.pth"
# CONFIG_FILE="models/VarifocalNet_R-101-FPN/VarifocalNet_R-101-FPN.py"
# CHECKPOINT_FILE="models/VarifocalNet_R-101-FPN/latest.pth"
# CONFIG_FILE="mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py"
# CHECKPOINT_FILE="mmdetection/checkpoints/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth"
# CONFIG_FILE="mmdetection/configs/yolox/yolox_tiny_8xb8-300e_coco.py"
# CHECKPOINT_FILE="mmdetection/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
# CONFIG_FILE="mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py"
# CHECKPOINT_FILE="mmdetection/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
# CONFIG_FILE="models/Faster R-CNN_R-101-FPN/Faster R-CNN_R-101-FPN.py"
# CHECKPOINT_FILE="models/Faster R-CNN_R-101-FPN/latest.pth"
# CONFIG_FILE="models/Faster R-CNN_R-50-FPN_voc0712/Faster R-CNN_R-50-FPN_voc0712.py"
# CONFIG_FILE="models/Cascade R-CNN_R-101-FPN/Cascade R-CNN_R-101-FPN.py"
# CHECKPOINT_FILE="models/Cascade R-CNN_R-101-FPN/latest.pth"
CONFIG_FILE="models_debug/Cascade Mask R-CNN_R-101-FPN/Cascade Mask R-CNN_R-101-FPN.py"
CHECKPOINT_FILE="models_debug/Cascade Mask R-CNN_R-101-FPN/latest.pth"

ALPHA=2.55
STEPS=2
EPSILON=8
ATTACK="cospgd"  # "pgd", "fgsm", "cospgd", "none"

cd .. # expects object_detection/slurm to be the working directory

echo "#########  START ATTACK ###########"
echo "CHECKPOINT_FILE="${CHECKPOINT_FILE}
echo "CONFIG_FILE="${CONFIG_FILE}
echo "ALPHA="${ALPHA}
echo "STEPS="${STEPS}
echo "EPSILON="${EPSILON}
echo "ATTACK="${ATTACK}

python -m pudb adv_attack.py --config_file "${CONFIG_FILE}" --checkpoint_file "${CHECKPOINT_FILE}" --steps ${STEPS} --alpha ${ALPHA} --epsilon ${EPSILON} --attack ${ATTACK} 