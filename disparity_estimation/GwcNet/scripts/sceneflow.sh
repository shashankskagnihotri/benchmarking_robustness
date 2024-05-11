#!/usr/bin/env bash
set -x

# Change directory to the directory containing main.py

DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/sceneflow/gwcnet-gc/checkpoint_000015.ckpt"

python main.py --dataset sceneflow \
    --datapath $DATAPATH \
    --epochs 16 --lrepochs "10,12,14,16:2" \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc \
    --loadckpt $CHECKPOINTPATH
    # --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \