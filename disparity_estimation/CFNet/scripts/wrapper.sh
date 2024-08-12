#!/usr/bin/env bash

source ./dataset_path.sh
source ./checkpoint_path.sh

DATASET="$1"
COMMON_CORRUPTION="$2"
SEVERITY_LEVEL="$3"
CHECKPOINTPATH="$4"

if [[ -z "$CHECKPOINTPATH" ]]; then    # Check if the checkpoint path is provided. If not, get the default checkpoint path.
    CHECKPOINTPATH=$(get_checkpoint_path "$DATASET")
    if [[ -z "$CHECKPOINTPATH" ]]; then
        echo "Error: Default checkpoint path not found."
        exit 1
    fi
fi


if [[ -z "$COMMON_CORRUPTION" || -z "$SEVERITY_LEVEL" ]]; then
    echo "Error: Corruption type and severity level must be provided."
    exit 1
fi

DATAPATH=$(get_dataset_path "$DATASET" "$COMMON_CORRUPTION" "$SEVERITY_LEVEL")

if [[ -z "$DATAPATH" ]]; then
    echo "Error: Dataset path not found."
    exit 1
fi

python wrapper.py \
    --scenario commoncorruption \
    --commoncorruption $COMMON_CORRUPTION \
    --severity $SEVERITY_LEVEL \
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