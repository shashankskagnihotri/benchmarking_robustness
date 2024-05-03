#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=internimage_ade20k_train
#SBATCH --output=slurm/internimage_ade20k_train.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";

source activate openmmlab

cd mmsegmentation

python tools/train.py configs/internimage/upernet_internimage_t_512_160k_ade20k.py

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime