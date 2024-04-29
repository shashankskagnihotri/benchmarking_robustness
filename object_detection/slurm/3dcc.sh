#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de
#SBATCH -o ./logs/output.%a.out # STDOUT

module load devel/cuda/11.8
cd .. # expects object_detection/slurm to be the working directory

# Create depth info
mv -n /data/coco/val2017/*.jpg DPT/input # move images
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt  # download model for depth estimation
python DPT/run_monodepth.py # assumes requirements are already installed
mkdir -p /data/coco/val2017/depth # create target folder 
mv DPT/input/*.png /data/coco/val2017/depth # move to taget folder

## Create 3D corruptions
cd 3DCommonCorruptions/create_3dcc 
PATH_RGB="../../data/coco/val2017"
PATH_DEPTH="../../data/coco/val2017"
PATH_TARGET="../../data/coco/3dcc"
python create_3dcc.py --path_rgb $PATH_RGB --path_depth $PATH_DEPTH --path_target $PATH_TARGET --batch_size 50 