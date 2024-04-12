#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de
#SBATCH -o ./logs/output.%a.out # STDOUT

CHECKPOINT_FILE="mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
CONFIG_FILE="mmdetection/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"
RESULT_FILE="results/robustness.out"

python tools/analysis_tools/test_robustness.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] --corruptions noise
