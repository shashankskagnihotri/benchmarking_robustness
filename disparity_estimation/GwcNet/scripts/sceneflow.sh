#!/usr/bin/env bash
set -x

# Change directory to the directory containing main.py
cd /pfs/work7/workspace/scratch/ma_adackerm-team_project_fss2024/benchmarking_robustness/disparity_estimation/GwcNet/

DATAPATH="/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/"
python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 16 --lrepochs "10,12,14,16:2" \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc