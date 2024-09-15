#!/usr/bin/env bash

DATAPATH="/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/KITTI_2015/Common_corruptions/no_corruption/severity_0"
# CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/CFNet/checkpoints/sceneflow_pretraining.ckpt"

python -m pudb wrapper.py --model cfnet \
    --scenario train \
    --dataset kitti2015 \
    --datapath $DATAPATH \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model cfnet \
    --logdir ./checkpoints/kitti/  \
    --test_batch_size 1 \
    #--debug True \
    # --loadckpt $CHECKPOINTPATH \
    