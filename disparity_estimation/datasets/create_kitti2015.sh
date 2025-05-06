#!/bin/bash

mkdir -p KITTI_2015/Common_corruptions/no_corruption/severity_0
mv testing KITTI_2015/Common_corruptions/no_corruption/severity_0
mv training KITTI_2015/Common_corruptions/no_corruption/severity_0

cd ../common_corruptions
python create_common_corruptions_kitti.py