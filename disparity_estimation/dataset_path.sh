#!/bin/bash

# Funktion zum Erstellen des Dataset-Pfads
get_dataset_path() {
    local dataset=$1
    local corruption_type=$2
    local severity_level=$3
    
    if [ -z "$corruption_type" ]; then
        corruption_type="no_corruption"
    fi

    case "$dataset" in
        "sceneflow")
            echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D/Common_corruptions/$corruption_type/severity_$severity_level"
            ;;
        
        "mpisintel")
            echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/mpisintel/Common_corruptions/brightness/severity_$severity_level"
            ;;
        
        "kitti2015")
            echo "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/KITTI_2015/Common_corruptions/$corruption_type/severity_$severity_level"
            ;;
        
        *)
            echo "Unknown dataset: $dataset" >&2
            exit 1
            ;;
    esac
}
