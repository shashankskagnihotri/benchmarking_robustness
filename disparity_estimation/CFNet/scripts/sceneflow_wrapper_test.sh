#!/usr/bin/env bash

DATAPATH="/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/cfnet/sceneflow_pretraining.ckpt"

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