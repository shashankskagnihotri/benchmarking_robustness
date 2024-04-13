#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint jhkjhk\
                --num_workers 2\
                --eval\
                --dataset kitti2015\
                --dataset_directory sample_data/KITTI_2015\
                --resume kitti_finetuned_model.pth.tar

python main.py  --batch_size 1\
                --checkpoint sdfsad\
                --num_workers 2\
                --eval\
                --dataset middlebury\
                --dataset_directory sample_data/MIDDLEBURY_2014\
                --resume kitti_finetuned_model.pth.tar