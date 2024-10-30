#!/usr/bin/env bash

source ./dataset_path.sh
source ./checkpoint_path.sh

# Default-Werte
DATASET=""
COMMON_CORRUPTION="no_corruption"
SEVERITY_LEVEL="0"
CHECKPOINTPATH=""
EXPERIMENT_NAME=""

MODEL="sttr"

# Parameter einlesen
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --attack_type) ATTACK_TYPE="$2"; shift ;;
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

if [[ -z "$ATTACK_TYPE" ]]; then
    echo "Error: Attack type must be provided."
    exit 1
fi


if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "Error: --experiment_name must be provided."
    exit 1
fi

#DATAPATH=$(get_dataset_path "$DATASET" "$COMMON_CORRUPTION" "$SEVERITY_LEVEL")

DATAPATH="/ceph/sagnihot/projects/benchmarking_robustness/disparity_estimation/data/FlyingThings3D"
CHECKPOINTPATH="/ceph/sagnihot/projects/benchmarking_robustness/disparity_estimation/pretrained_weights/sceneflow_pretrained_model.pth.tar"

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
    --model "$MODEL" \
    --logdir "./checkpoints/$DATASET/uniform_sample_d256"  \
    --test_batch_size 1