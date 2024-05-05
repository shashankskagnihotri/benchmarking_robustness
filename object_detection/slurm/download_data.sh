#!/bin/bash
#SBATCH --partition=single
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de
#SBATCH -o ./logs/%j.out # STDOUT

module load devel/cuda/11.8 # needed
cd ..

# voc2007
DATASET="voc2007"
DATA_PATH="data/"

python mmdetection/tools/misc/download_dataset.py --dataset-name $DATASET --save-dir $DATA_PATH --unzip --delete

# voc2012
DATASET="voc2012"
DATA_PATH="data/"

python mmdetection/tools/misc/download_dataset.py --dataset-name $DATASET --save-dir $DATA_PATH --unzip --delete

# # lvis
# DATASET="lvis"
# DATA_PATH="data/"$DATASET

# python mmdetection/tools/misc/download_dataset.py --dataset-name $DATASET --save-dir $DATA_PATH --unzip --delete

# # coco
# DATASET="coco"
# DATA_PATH="data/"$DATASET

# python mmdetection/tools/misc/download_dataset.py --dataset-name $DATASET --save-dir $DATA_PATH --unzip --delete