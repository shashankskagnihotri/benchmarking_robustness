#!/usr/bin/env bash
set -x
DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/eth3d/"
python save_disp_eth3d.py \
    --datapath $DATAPATH \
    --testlist ./filenames/kitti15_test.txt \
    --model cfnet \
    --dataset eth3d \
    --maxdisp 256 \
    --loadckpt "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/cfnet/finetuning_model.ckpt"