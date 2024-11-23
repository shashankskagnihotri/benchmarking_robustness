#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=corruption_segformer_mit-b2_8xb2-160k_voc12aug-512x512
#SBATCH --output=slurm/corruption_segformer_mit-b2_8xb2-160k_voc12aug-512x512.out
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
    python tools/test.py ../corruptions/corruption_config/pascalvoc/segformer/segformer_mit-b2_8xb2-160k_voc12aug-512x512.py \
        ../checkpoint_files/pascalvoc/segformer/segformer_mit-b2_160k.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/pascalvoc/segformer/segformer_mit-b2_8xb2-160k_voc12aug-512x512/$corruption

    echo "Finished processing corruption type: $corruption"
done

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"
