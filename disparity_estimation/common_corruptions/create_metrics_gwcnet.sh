#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --time=24:00:00
#SBATCH --job-name=Create_Metrics_Common_corruptions_SceneFlow_CFNet

source ~/.bashrc
conda activate unified_env

cd /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation

# Liste der Bildkorruptionen
corruptions=(
    'gaussian_noise'
    'shot_noise'
    'impulse_noise'
    'defocus_blur'
    'glass_blur'
    'motion_blur'
    'zoom_blur'
    'snow'
    'frost'
    'fog'
    'brightness'
    'contrast'
    'elastic_transform'
    'pixelate'
    'jpeg_compression'
)

# Dataset und Experimentname
dataset="sceneflow"
experiment_name="Common_Corruptions"

# Schleife über jede Korruption
for corruption in "${corruptions[@]}"; do
    # Schleife über die Severity-Level von 0 bis 5
    for severity in {0..4}; do
        # Befehl ausführen
        echo "Running: /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/GwcNet/scripts/wrapper_corruptions_test.sh --dataset $dataset --common_corruption $corruption --severity $severity --experiment_name $experiment_name"
        /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/code/benchmarking_robustness/disparity_estimation/GwcNet/scripts/wrapper_corruptions_test.sh --dataset "$dataset" --common_corruption "$corruption" --severity "$severity" --experiment_name "$experiment_name"
    done
done