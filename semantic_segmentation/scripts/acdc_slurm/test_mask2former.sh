#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test_mask2former_all_combinations.out
#SBATCH --output=slurm/test_mask2former_%a.out
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]] 
then
    python tools/test.py ../configs/acdc_configs/mask2former_r50_8xb2-90k_acdc-512x1024.py ../checkpoint_files/mask2former/mask2former_r50_8xb2-90k_acdc.pth --work-dir ../acdc_work_dir/mask2former/mask2former_r50_8xb2-90k_acdc-512x1024  --show-dir ../acdc_work_dir/mask2former/mask2former_r50_8xb2-90k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]  
then
    python tools/test.py ../configs/acdc_configs/mask2former_r101_8xb2-90k_acdc-512x1024.py ../checkpoint_files/mask2former/mask2former_r101_8xb2-90k_acdc.pth --work-dir ../acdc_work_dir/mask2former/mask2former_r101_8xb2-90k_acdc-512x1024 --show-dir ../acdc_work_dir/mask2former/mask2former_r101_8xb2-90k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]] 
then
    python tools/test.py ../configs/acdc_configs/mask2former_swin-s_8xb2-90k_acdc-512x1024.py ../checkpoint_files/mask2former/mask2former_swin-s_8xb2-90k_acdc.pth --work-dir ../acdc_work_dir/mask2former/mask2former_swin-s_8xb2-90k_acdc-512x1024 --show-dir ../acdc_work_dir/mask2former/mask2former_swin-s_8xb2-90k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]] 
then
    python tools/test.py ../configs/acdc_configs/mask2former_swin-t_8xb2-90k_acdc-512x1024.py ../checkpoint_files/mask2former/mask2former_swin-t_8xb2-90k_acdc.pth --work-dir ../acdc_work_dir/mask2former/mask2former_swin-t_8xb2-90k_acdc-512x1024 --show-dir ../acdc_work_dir/mask2former/mask2former_swin-t_8xb2-90k_acdc-512x1024/show_dir
else
    echo "All submitted!"
fi


end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime

