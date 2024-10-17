#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=unet-s5-d16_fcn_4x4-40k_pascal_voc-256-256
#SBATCH --output=slurm/unet-s5-d16_fcn_4x4-40k_pascal_voc-256-256_train.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de
echo "Started at $(date)";

source activate semseg
cd mmsegmentation

python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-40k_pascal_voc-256x256.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-40k_pascal_voc-256x256


end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime