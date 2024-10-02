#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=corruption_upernet_internimage_h_896_160k_ade20k
#SBATCH --output=slurm/corruption_upernet_internimage_h_896_160k_ade20k.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310
cd mmsegmentation

# List of corruption methods
corruptions=(
    'gaussian_noise'
    'shot_noise'
    'impulse_noise'
    'defocus_blur'
    'glass_blur'
    'motion_blur'
    'zoom_blur'
    'snow'
    'frost'
    'fog'
    'brightness'
    'contrast'
    'elastic_transform'
    'pixelate'
    'jpeg_compression'
    'elastic_transform'
    'pixelate'
    'jpeg_compression'
)

# Loop through each corruption type and run the Python command
for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/ade20k/internimage/upernet_internimage_h_896_160k_ade20k.py \
        ../checkpoint_files/ade20k/internimage/upernet_internimage_h_896_160k_ade20k.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/ade20k/internimage/upernet_internimage_h_896_160k_ade20k/$corruption

    echo "Finished processing corruption type: $corruption"
done

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"
