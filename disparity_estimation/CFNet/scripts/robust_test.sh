#!/usr/bin/env bash
set -x
DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/CFNet/checkpoints/sceneflow_pretraining.ckpt"
CUDA_VISIBLE_DEVICES=0 python robust_test.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_errortest.txt --batch_size 4 --test_batch_size 2 \
    --testlist ./filenames/kitti15_errortest.txt --maxdisp 256 \
    --epochs 1 --lr 0.001  --lrepochs "300:10" \
    --loadckpt $CHECKPOINTPATH \
    --model cfnet --logdir ./checkpoints/robust_abstudy_test
