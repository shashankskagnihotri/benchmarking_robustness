#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python wrapper.py  --batch_size 1\
                --scenario train \
                --model sttr \
                --num_workers 4\
                --dataset mpisintel\
                --dataset_directory /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/mpisintel/Common_corruptions/no_corruption/severity_0/
