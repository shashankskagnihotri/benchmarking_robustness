#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint sdfsad\
                --num_workers 4\
                --eval\
                --dataset sceneflow\
                --dataset_directory /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D\
                --resume /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/stereo-transformer/sceneflow_pretrained_model.pth.tar

