#!/usr/bin/env bash
set -x

# Change directory to the directory containing main.py
cd /pfs/work7/workspace/scratch/ma_adackerm-team_project_fss2024/benchmarking_robustness/disparity_estimation/GwcNet/

DATAPATH="/pfs/work7/workspace/scratch/ma_adackerm-team_project_fss2024/dataset/KITTI_2015/"
python -W ignore main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-g --logdir ./checkpoints/kitti15/gwcnet-g --loadckpt ./checkpoints/kitti15/gwcnet-g/best.ckpt \
    --test_batch_size 1  --eval