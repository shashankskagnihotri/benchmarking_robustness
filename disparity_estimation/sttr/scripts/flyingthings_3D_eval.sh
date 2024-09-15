#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0

cd /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/code/benchmarking_robustness/disparity_estimation/
python wrapper.py  --batch_size 1\
                --scenario train \
                --model sttr \
                --num_workers 4\
                --dataset sceneflow\
                --dataset_directory /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0 \
                #--validation train
                #--resume /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/sttr/sceneflow_pretrained_model.pth.tar
                

