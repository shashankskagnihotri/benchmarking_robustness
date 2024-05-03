#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=first_ade20k_train
#SBATCH --output=slurm/first_ade20k_train.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";

source activate openmmlab

cd mmsegmentation

python tools/train.py configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512.py

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime