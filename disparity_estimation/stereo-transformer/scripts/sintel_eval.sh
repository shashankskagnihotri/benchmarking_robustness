#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint sintel\
                --num_workers 4\
                --eval\
                --dataset sintel\
                --dataset_directory /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/mpi_sintel_stero/train/ \
                --resume sceneflow_pretrained_model.pth.tar

