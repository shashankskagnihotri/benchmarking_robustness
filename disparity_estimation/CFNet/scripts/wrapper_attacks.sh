#!/usr/bin/env bash

source ./dataset_path.sh
source ./checkpoint_path.sh

DATASET="$1"
ATTACK_TYPE="$2"
SEVERITY_LEVEL="0"
CHECKPOINTPATH="$3"
COMMONCORRUPTION="no_corruption"

if [[ -z "$DATASET" ]]; then
    echo "Error: Dataset must be provided."
    exit 1
fi

if [[ -z "$CHECKPOINTPATH" ]]; then    # Check if the checkpoint path is provided. If not, get the default checkpoint path.
    CHECKPOINTPATH=$(get_checkpoint_path "$DATASET")
    if [[ -z "$CHECKPOINTPATH" ]]; then
        echo "Error: Default checkpoint path not found."
        exit 1
    fi
fi


if [[ -z "$ATTACK_TYPE" ]]; then
    echo "Error: Attack type must be provided."
    exit 1
fi

DATAPATH=$(get_dataset_path "$DATASET" "$COMMON_CORRUPTION" "$SEVERITY_LEVEL")

if [[ -z "$DATAPATH" ]]; then
    echo "Error: Dataset path not found."
    exit 1
fi

python wrapper.py \
    --scenario attack \
    --attack_type $ATTACK_TYPE \
    --dataset $DATASET \
    --datapath $DATAPATH \
    --loadckpt $CHECKPOINTPATH \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model cfnet \
    --logdir "./checkpoints/$DATASET/uniform_sample_d256"  \
    --test_batch_size 1