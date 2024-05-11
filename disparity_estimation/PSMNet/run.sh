#!/bin/bash

DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D"

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath $DATAPATH \
               --epochs 1 \
               --loadmodel /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/psmnet/pretrained_sceneflow_new.tar \
               --savemodel ./trained/ \
               --dataset sceneflow



# python finetune.py --maxdisp 192 \
#                    --model stackhourglass \
#                    --datatype 2015 \
#                    --datapath $DATAPATH \
#                    --epochs 300 \
#                    --loadmodel /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/psmnet/pretrained_sceneflow_new.tar \
#                    --savemodel ./trained/\
#                    --dataset sceneflow

