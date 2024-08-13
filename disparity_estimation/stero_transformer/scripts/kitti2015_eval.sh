#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint jhkjhk\
                --num_workers 2\
                --eval\
                --dataset kitti2015\
                --dataset_directory /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/KITTI_2015\
                --resume /pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/pretrained_weights/stereo-transformer/kitti_finetuned_model.pth.tar

# python main.py  --batch_size 1\
#                 --checkpoint sdfsad\
#                 --num_workers 2\
#                 --eval\
#                 --dataset middlebury\
#                 --dataset_directory sample_data/MIDDLEBURY_2014\
#                 --resume kitti_finetuned_model.pth.tar