#!/bin/bash

# Funktion zum Erstellen des Dataset-Pfads
get_checkpoint_path() {
    local dataset=$1
    local model=$2
    
    if [ -z "$dataset" ]; then
        echo "Unknown dataset" >&2
        exit 1
    fi

    if [ -z "$model" ]; then
        echo "Unknown model" >&2
        exit 1
    fi


    case "$dataset" in
        "sceneflow")
        
            case "$model" in
                "gwcnet-gc")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/sceneflow/gwcnet-gc/checkpoint_000015.ckpt"
                    ;;

                "gwcnet-g")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/sceneflow/gwcnet-g/checkpoint_000015.ckpt"
                    ;;
                    
                "cfnet")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/cfnet/sceneflow_pretraining.ckpt"
                    ;;
                "sttr")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/sttr/sceneflow_pretrained_model.pth.tar"
                    ;;
                "sttr-light")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/sttr-light/sttr_light_sceneflow_pretrained_model.pth.tar"
                    ;;
                *)
                    echo "Unknown model: $model" >&2
                    exit 1
                    ;;
            esac
            ;;

        "kitti2015")
            case "$model" in
                "gwcnet-gc")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/kitti15/gwcnet-g/best.ckpt"
                    ;;

                "gwcnet-g")
                    echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/gwcnet/kitti15/gwcnet-g/best.ckpt"
                    ;;
                    
                *)
                    echo "Unknown model: $model" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            exit 1
            ;;
    esac
}
