#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=internimage_upernet_internimage_t_160k_voc12aug_512x512_train
#SBATCH --output=slurm/internimage_upernet_internimage_t_160k_voc12aug_512x512_train.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";

source activate py310

cd mmsegmentation

python tools/train.py ../configs/internimage/upernet_internimage_t_160k_voc12aug_512x512.py --work-dir ../work_dirs/upernet_internimage_t_160k_voc12aug_512x512.py

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime