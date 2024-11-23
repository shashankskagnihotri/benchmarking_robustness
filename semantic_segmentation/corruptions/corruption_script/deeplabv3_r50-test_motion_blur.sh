#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=corruption_deeplabv3_r50-d8_4xb4-40k_voc12aug-512x512
#SBATCH --output=slurm/corruption_deeplabv3_r50-d8_4xb4-40k_voc12aug-512x512.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310v2
cd mmsegmentation

# Loop through each corruption type and run the Python command

echo "Processing corruption type: $corruption"
python tools/test.py ../corruptions/corruption_config/pascalvoc/deeplabv3/deeplabv3_r50-d8_4xb4-40k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_20200613_161546-2ae96e7e.pth \
        --cfg-options "model.data_preprocessor.corruption='motion_blur'" \
        --work-dir ../corruptions/work_dirs/pascalvoc/deeplabv3/deeplabv3_test_voc12aug-512x512/motion_blur

echo "Finished processing corruption type: motion_blur"


end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"
