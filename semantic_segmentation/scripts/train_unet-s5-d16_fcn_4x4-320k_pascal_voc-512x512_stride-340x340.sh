#!/usr/bin/env bash
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_unet-s5-d16_fcn_4x4-320k_pascal_voc-512x512_stride-340x340
#SBATCH --output=slurm/train_unet-s5-d16_fcn_4x4-320k_pascal_voc-512x512_stride-340x340.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-320k_pascal_voc-512x512_stride-340x340.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-320k_pascal_voc-512x512_stride-340x340

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime
