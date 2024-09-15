#!/usr/bin/env bash

source ./dataset_path.sh
source ./checkpoint_path.sh

# Default-Werte
DATASET=""
COMMON_CORRUPTION=""
SEVERITY_LEVEL=""
CHECKPOINTPATH=""
EXPERIMENT_NAME=""

MODEL="cfnet"

# Parameter einlesen
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --common_corruption) COMMON_CORRUPTION="$2"; shift ;;
        --severity) SEVERITY_LEVEL="$2"; shift ;;
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

if [[ -z "$COMMON_CORRUPTION" || -z "$SEVERITY_LEVEL" ]]; then
    echo "Error: --common_corruption and --severity must be provided."
    exit 1
fi

if [[ -z "$CHECKPOINTPATH" ]]; then
    CHECKPOINTPATH=$(get_checkpoint_path "$DATASET" "$MODEL")
    if [[ -z "$CHECKPOINTPATH" ]]; then
        echo "Error: Default checkpoint path not found."
        exit 1
    fi
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "Error: --experiment_name must be provided."
    exit 1
fi

DATAPATH=$(get_dataset_path "$DATASET" "$COMMON_CORRUPTION" "$SEVERITY_LEVEL")

if [[ -z "$DATAPATH" ]]; then
    echo "Error: Dataset path not found."
    exit 1
fi

# Python-Skript ausführen
python wrapper.py \
    --scenario commoncorruption \
    --commoncorruption "$COMMON_CORRUPTION" \
    --severity "$SEVERITY_LEVEL" \
    --dataset "$DATASET" \
    --datapath "$DATAPATH" \
    --loadckpt "$CHECKPOINTPATH" \
    --epochs 20 \
    --lr 0.001 \
    --lrepochs "12,16,18,20:2" \
    --batch_size 1 \
    --maxdisp 256 \
    --model sttr \
    --logdir "./checkpoints/$DATASET/uniform_sample_d256" \
    --test_batch_size 1 \
    --experiment "$EXPERIMENT_NAME"
