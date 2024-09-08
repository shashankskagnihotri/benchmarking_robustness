#!/usr/bin/env bash

source ./dataset_path.sh

COMMON_CORRUPTION="$1"
SEVERITY_LEVEL="$2"
EXPERIMENT_NAME="$3"

DATAPATH=$(get_dataset_path "sceneflow" "$COMMON_CORRUPTION" "$SEVERITY_LEVEL")
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/cfnet/sceneflow_pretraining.ckpt"

if [[ -z "$COMMON_CORRUPTION" || -z "$SEVERITY_LEVEL" ]]; then
    echo "Error: Corruption type and severity level must be provided."
    exit 1
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "Error: Experiment name must be provided."
    exit 1
fi

python wrapper.py \
    --scenario commoncorruption \
    --commoncorruption $COMMON_CORRUPTION \
    --severity $SEVERITY_LEVEL \
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
    --test_batch_size 1 \
    --experiment $EXPERIMENT_NAME