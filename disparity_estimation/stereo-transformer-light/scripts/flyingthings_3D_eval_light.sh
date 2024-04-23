#!/usr/bin/env bash

python main.py  --batch_size 1\
                --checkpoint sdfsad\
                --num_workers 4\
                --eval\
                --dataset sceneflow\
                --dataset_directory /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D\
                --resume sttr_light_sceneflow_pretrained_model.pth.tar