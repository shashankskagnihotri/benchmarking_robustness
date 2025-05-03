#!/bin/bash

cd FlyingThings3D
mkdir -p Common_corruptions/no_corruption/severity_0
mv frames_finalpass Common_corruptions/no_corruption/severity_0

cd ../../common_corruptions
python create_common_corruptions_sceneflow.py