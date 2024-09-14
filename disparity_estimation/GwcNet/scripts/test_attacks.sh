#!/usr/bin/env bash

DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D/"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/sceneflow/gwcnet-gc/checkpoint_000015.ckpt"

ATTACK_TYPE="$1"

echo "Attack type: $ATTACK_TYPE"

if [[ -z "$ATTACK_TYPE" ]]; then
    echo "Error: Attack type must be provided."
    exit 1
fi

python wrapper.py --architecture gwcnet-gc \
    --scenario attack \
    --attack_type $ATTACK_TYPE \
    --dataset sceneflow \
    --datapath $DATAPATH \
    --loadckpt $CHECKPOINTPATH \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model gwcnet \
    --logdir ./checkpoints/sceneflow/uniform_sample_d256  \
    --test_batch_size 1