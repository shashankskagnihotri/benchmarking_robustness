#!/usr/bin/env bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=corruption_pascal_motion_blurs
#SBATCH --output=slurm/corruption_pascal_motion_blurs.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310
cd mmsegmentation

# List of corruption methods
corruptions=(
    'motion_blur'
)

# Loop through each corruption type and run the Python command
for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/pascalvoc/deeplabv3/deeplabv3_r50-d8_4xb4-40k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_20200613_161546-2ae96e7e.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/pascalvoc/deeplabv3/deeplabv3_r50-d8_4xb4-40k_voc12aug-512x512/$corruption

    echo "Finished processing corruption type: $corruption"
done

# Loop through each corruption type and run the Python command
for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/pascalvoc/deeplabv3/deeplabv3_r101-d8_4xb4-40k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/deeplabv3/deeplabv3_r101-d8_512x512_40k_voc12aug_20200613_161432-0017d784.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/pascalvoc/deeplabv3/deeplabv3_r101-d8_4xb4-40k_voc12aug-512x512/$corruption

    echo "Finished processing corruption type: $corruption"
done


# Loop through each corruption type and run the Python command
for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/pascalvoc/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/pascalvoc/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512/$corruption

    echo "Finished processing corruption type: $corruption"
done

# Loop through each corruption type and run the Python command
for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/pascalvoc/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/pascalvoc/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512/$corruption

    echo "Finished processing corruption type: $corruption"
done

# Loop through each corruption type and run the Python command
for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/pascalvoc/pspnet/pspnet_r50-d8_4xb4-40k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/pspnet/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/pascalvoc/pspnet/pspnet_r50-d8_4xb4-40k_voc12aug-512x512/$corruption

    echo "Finished processing corruption type: $corruption"
done



end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"
