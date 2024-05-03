#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=internimage_b_pascaltrain
#SBATCH --output=slurm/internimage_b_train.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";

source activate openmmlab

cd mmsegmentation

python tools/train.py ../configs/internimage/upernet_internimage_b_160k_pascal_2048x512.py --work-dir ../work_dirs/upernet_internimage_b_160k_pascal_2048x512

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime
