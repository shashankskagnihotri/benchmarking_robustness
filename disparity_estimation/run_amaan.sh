#!/bin/bash

source ./dataset_path.sh

dataset_path=$(get_dataset_path "kitti2015" "no_corruption" "0")

python -m pudb wrapper.py --model sttr --scenario test --dataset kitti --experiment debug --datapath "$dataset_path"