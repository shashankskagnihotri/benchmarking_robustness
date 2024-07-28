#!/usr/bin/env bash
cd /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/code/benchmarking_robustness/disparity_estimation/
DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D/"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/CFNet/checkpoints/sceneflow_pretraining.ckpt"


python wrapper.py --model cfnet \
    --scenario test \
    --dataset sceneflow \
    --datapath $DATAPATH \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model cfnet \
    --loadckpt $CHECKPOINTPATH \
    --logdir ./checkpoints/sceneflow/uniform_sample_d256  \
    --test_batch_size 1