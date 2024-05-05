#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de
#SBATCH -o ./logs/%j.out # STDOUT

module load devel/cuda/11.8
cd .. # expects object_detection/slurm to be the working directory

WEIGHTS_DIR="DPT/weights"
WEIGHTS_FILE="dpt_hybrid-midas-501f0c75.pt"
WEIGHTS_URL="https://github.com/intel-isl/DPT/releases/download/1_0/${WEIGHTS_FILE}"

cd $WEIGHTS_DIR
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Weight file does not exist, downloading..."
    wget $WEIGHTS_URL # Download the weights file
else
    echo "Weight file already exists, no need to download."
fi

# Move back to the original directory
cd ../..
create_depth_info() {
    local PATH_RGB=$1
    local PATH_DEPTH=$2

    echo "Starting depth information creation..."
    echo "Moving images to DPT input directory..."
    mv -n $PATH_RGB/*.jpg DPT/input
    cd DPT/
    echo "Running depth estimation model..."
    python run_monodepth.py
    cd ..
    echo "Creating target directory for depth information..."
    mkdir -p $PATH_DEPTH
    echo "Moving depth output to target directory..."
    mv DPT/output_monodepth/*.png $PATH_DEPTH
    echo "Restoring original images..."
    mv DPT/input/*.jpg $PATH_RGB
    echo "Depth information creation completed."
}

create_3d_corruptions() {
    local PATH_RGB="../.."$1
    local PATH_DEPTH="../.."$2
    local PATH_TARGET="../.."$3

    echo "Starting 3D corruption creation..."
    cd 3DCommonCorruptions/create_3dcc 
    echo "Running 3D corruption script..."
    python create_3dcc.py --path_rgb $PATH_RGB --path_depth $PATH_DEPTH --path_target $PATH_TARGET --batch_size 1
    echo "3D corruption creation completed."
    cd ../..
}

# COCO dataset
PATH_DEPTH="data/coco/val2017_depth"
PATH_RGB="data/coco/val2017/val2017"
PATH_TARGET="data/coco/3dcc"

echo "Processing COCO dataset..."
# create_depth_info $PATH_RGB $PATH_DEPTH
create_3d_corruptions $PATH_RGB $PATH_DEPTH $PATH_TARGET
echo "COCO dataset processing completed."

# PASCAL VOC 2007
PATH_DEPTH="data/VOCdevkit/VOC2007/depth"
PATH_RGB="data/VOCdevkit/VOC2007/JPEGImages"
PATH_TARGET="data/VOCdevkit/VOC2007/3dcc"

echo "Processing Pascal VOC 2007 dataset..."
# create_depth_info $PATH_RGB $PATH_DEPTH
create_3d_corruptions $PATH_RGB $PATH_DEPTH $PATH_TARGET
echo "Pascal VOC 2007 dataset processing completed."

# PASCAL VOC 2012
PATH_DEPTH="data/VOCdevkit/VOC2012/depth"
PATH_RGB="data/VOCdevkit/VOC2012/JPEGImages"
PATH_TARGET="data/VOCdevkit/VOC2012/3dcc"

echo "Processing Pascal VOC 2012 dataset..."
# create_depth_info $PATH_RGB $PATH_DEPTH
create_3d_corruptions $PATH_RGB $PATH_DEPTH $PATH_TARGET
echo "Pascal VOC 2012 dataset processing completed."