#!/bin/bash

mkdir -p FlyingThings3D
cp all_unused_files.txt FlyingThings3D
cd FlyingThings3D

URLS=(
    "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar"
    "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2"
    "https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_disparity_occlusions.tar.bz2"
)

for url in "${URLS[@]}"; do
    echo "Downloading $url..."
    wget -c "$url"
done

tar -xvjf FlyingThings3D_subset_disparity_occlusions.tar.bz2
tar -xvjf flyingthings3d__disparity.tar.bz2
tar -xvf flyingthings3d__frames_finalpass.tar

rm FlyingThings3D_subset_disparity_occlusions.tar.bz2
rm flyingthings3d__disparity.tar.bz2
rm flyingthings3d__frames_finalpass.tar

mkdir -p occlussion/TRAIN
mkdir -p occlussion/TEST
mv FlyingThings3D_subset/train/disparity_occlusions occlussion/TRAIN
mv FlyingThings3D_subset/val/disparity_occlusions occlussion/TEST
rm -r FlyingThings3D_subset