#!/bin/bash

# Funktion zum Erstellen des Dataset-Pfads
get_checkpoint_path() {
    local dataset=$1
    
    if [ -z "$dataset" ]; then
        echo "Unknown dataset" >&2
        exit 1
    fi

    case "$dataset" in
        "sceneflow")
            echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/cfnet/sceneflow_pretraining.ckpt"
            ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            exit 1
            ;;
    esac
}
