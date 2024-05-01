#!/bin/bash

DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D"

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath $DATAPATH \
               --epochs 0 \
               --loadmodel ./pretrained_weights/pretrained_sceneflow_new.tar \
               --savemodel ./trained/



# python finetune.py --maxdisp 192 \
                #    --model stackhourglass \
                #    --datatype 2015 \
                #    --datapath dataset/data_scene_flow_2015/training/ \
                #    --epochs 300 \
                #    --loadmodel ./trained/checkpoint_10.tar \
                #    --savemodel ./trained/

