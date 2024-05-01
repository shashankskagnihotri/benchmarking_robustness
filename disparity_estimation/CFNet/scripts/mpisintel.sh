#!/usr/bin/env bash
DATAPATH="/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/mpi_sintel_stero/train"
CHECKPOINTPATH="/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/code/benchmarking_robustness.worktrees/disparity_estimation/dataloader/disparity_estimation/CFNet/checkpoints/sceneflow_pretraining.ckpt"

python main.py --dataset sintel \
    --datapath $DATAPATH \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 1 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/sceneflow/uniform_sample_d256  \
    --test_batch_size 1