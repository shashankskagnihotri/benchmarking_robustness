#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test_deeplabv3plus_all_combinations.out
#SBATCH --output=slurm/test_deeplabv3plus_%a.out
#SBATCH --array=1-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]] 
then
    python tools/test.py ../configs/acdc_configs/deeplabv3plus_r50-d8_4xb2-80k_acdc-512x1024.py ../checkpoint_files/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_acdc.pth --work-dir ../acdc_work_dir/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_acdc-512x1024 --show-dir ../acdc_work_dir/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]  
then
    python tools/test.py ../configs/acdc_configs/deeplabv3plus_r101-d8_4xb2-80k_acdc-512x1024.py ../checkpoint_files/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_acdc.pth --work-dir ../acdc_work_dir/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_acdc-512x1024 --show-dir ../acdc_work_dir/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_acdc-512x1024/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]] 
then
    python tools/test.py ../configs/acdc_configs/deeplabv3plus_r18-d8_4xb2-80k_acdc-512x1024.py ../checkpoint_files/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_acdc.pth --work-dir ../acdc_work_dir/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_acdc-512x1024 --show-dir ../acdc_work_dir/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_acdc-512x1024/show_dir
else
    echo "All submitted!"
fi


end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime

