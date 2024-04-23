#!/usr/bin/env bash
set -x
DATAPATH="/pfs/work7/workspace/scratch/ma_adackerm-team_project_fss2024/dataset/KITTI_2015/"
python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model gwcnet-g --loadckpt ./checkpoints/kitti15/gwcnet-g/best.ckpt
