#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=first_ade20k_tests
#SBATCH --output=slurm/first_ade20k_tests.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";

source activate openmmlab
cd mmsegmentation

python tools/test.py configs/pspnet/pspnet_r50-d8_4xb4-160k_ade20k-512x512.py ../checkpoint_files/pspnet_r50-d8_512x512_160k_ade20k_20200615_184358-1890b0bd.pth
end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime