#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test_segformer_mit-b0_8xb1-160k_acdc-512x1024
#SBATCH --output=slurm/test_segformer_mit-b0_8xb1-160k_acdc-512x1024.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

python tools/test.py ../configs/acdc_configs/segformer_mit-b0_8xb1-160k_acdc-512x1024.py ../checkpoint_files/segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.pth --work-dir ../acdc_work_dir/segformer/segformer_mit-b0_8xb1-160k_acdc-512x1024 --show-dir ../acdc_work_dir/segformer/segformer_mit-b0_8xb1-160k_acdc-512x1024/show_dir

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime
