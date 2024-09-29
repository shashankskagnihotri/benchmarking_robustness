#!/usr/bin/env bash

source ./dataset_path.sh
source ./checkpoint_path.sh

# Default-Werte
DATASET=""
CHECKPOINTPATH=""
EXPERIMENT_NAME=""

MODEL="sttr-light"

# Parameter einlesen
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --checkpoint) CHECKPOINTPATH="$2"; shift ;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Fehlerbehandlung
if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset must be provided."
    exit 1
fi

if [[ -z "$CHECKPOINTPATH" ]]; then    # Check if the checkpoint path is provided. If not, get the default checkpoint path.
    CHECKPOINTPATH=""
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "Error: --experiment_name must be provided."
    exit 1
fi

DATAPATH=$(get_dataset_path "$DATASET" "no_corruption" "0")

if [[ -z "$DATAPATH" ]]; then
    echo "Error: Dataset path not found."
    exit 1
fi

python wrapper.py \
    --scenario train \
    --dataset $DATASET \
    --datapath $DATAPATH \
    --loadckpt $CHECKPOINTPATH \
    --epochs 400 \
    --ft \
    --model "$MODEL" \
    --logdir "./checkpoints/$DATASET/uniform_sample_d256"