#!/usr/bin/env bash
# set -x

# Change directory to the directory containing main.py
#cd /pfs/work7/workspace/scratch/ma_adackerm-team_project_fss2024/benchmarking_robustness/disparity_estimation/GwcNet/

DATAPATH="/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/KITTI_2015/"
python -W ignore main.py --dataset kitti \
    --datapath $DATAPATH  \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-g --logdir ./checkpoints/kitti15/gwcnet-g \
    --loadckpt /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/sceneflow/gwcnet-g/checkpoint_000015.ckpt \
    --test_batch_size 1  # --eval