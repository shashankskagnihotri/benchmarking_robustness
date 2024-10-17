#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test_unet-s5-d16_fcn_4xb4-160k_acdc-512x1024
#SBATCH --output=slurm/test_unet-s5-d16_fcn_4xb4-160k_acdc-512x1024.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

python tools/test.py ../configs/acdc_configs/unet-s5-d16_fcn_4xb4-160k_acdc-512x1024.py ../checkpoint_files/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --work-dir ../acdc_work_dir/unet/

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime
