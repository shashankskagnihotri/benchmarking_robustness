#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test_segformer_all_combinations.out
#SBATCH --output=slurm/test_segformer_mit-b%a_8xb1-160k_acdc-512x1024.out
#SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]] 
then
    python tools/test.py ../configs/acdc_configs/segformer_mit-b1_8xb1-160k_acdc-512x1024.py ../checkpoint_files/segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.pth --work-dir ../acdc_work_dir/segformer/segformer_mit-b1_8xb1-160k_acdc-512x1024 --show-dir ../acdc_work_dir/segformer/segformer_mit-b1_8xb1-160k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]  
then
    python tools/test.py ../configs/acdc_configs/segformer_mit-b2_8xb1-160k_acdc-512x1024.py ../checkpoint_files/segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.pth --work-dir ../acdc_work_dir/segformer/segformer_mit-b2_8xb1-160k_acdc-512x1024 --show-dir ../acdc_work_dir/segformer/segformer_mit-b2_8xb1-160k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]] 
then
    python tools/test.py ../configs/acdc_configs/segformer_mit-b3_8xb1-160k_acdc-512x1024.py ../checkpoint_files/segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.pth --work-dir ../acdc_work_dir/segformer/segformer_mit-b3_8xb1-160k_acdc-512x1024 --show-dir ../acdc_work_dir/segformer/segformer_mit-b3_8xb1-160k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]] 
then
    python tools/test.py ../configs/acdc_configs/segformer_mit-b4_8xb1-160k_acdc-512x1024.py ../checkpoint_files/segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.pth --work-dir ../acdc_work_dir/segformer/segformer_mit-b4_8xb1-160k_acdc-512x1024 --show-dir ../acdc_work_dir/segformer/segformer_mit-b4_8xb1-160k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]] 
then
    python tools/test.py ../configs/acdc_configs/segformer_mit-b5_8xb1-160k_acdc-512x1024.py ../checkpoint_files/segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.pth --work-dir ../acdc_work_dir/segformer/segformer_mit-b5_8xb1-160k_acdc-512x1024 --show-dir ../acdc_work_dir/segformer/segformer_mit-b5_8xb1-160k_acdc-512x1024/show_dir
else
    echo "All submitted!"
fi


end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime

