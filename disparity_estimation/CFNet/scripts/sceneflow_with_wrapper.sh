#!/usr/bin/env bash

DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/KITTI_2015/"

cd /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/code/benchmarking_robustness/disparity_estimation/

python wrapper.py -a cfnet -m train \
    --dataset kitti2015 \
    --datapath $DATAPATH \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model cfnet \
    --logdir /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/checkpoints/cfnet/  \
    --test_batch_size 1