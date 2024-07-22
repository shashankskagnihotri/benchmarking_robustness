#!/usr/bin/env bash

DATAPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D/$CORRUPTION_TYPE/$SEVERITY_LEVEL"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/CFNet/checkpoints/sceneflow_pretraining.ckpt"

CORRUPTION_TYPE="$1"
SEVERITY_LEVEL="$2"

if [[ -z "$CORRUPTION_TYPE" || -z "$SEVERITY_LEVEL" ]]; then
    echo "Error: Corruption type and severity level must be provided."
    exit 1
fi

python wrapper.py --architecture cfnet \
    --mode commoncorruption \
    --dataset sceneflow \
    --datapath $DATAPATH \
    --loadckpt $CHECKPOINTPATH \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model cfnet \
    --logdir ./checkpoints/sceneflow/uniform_sample_d256  \
    --test_batch_size 1