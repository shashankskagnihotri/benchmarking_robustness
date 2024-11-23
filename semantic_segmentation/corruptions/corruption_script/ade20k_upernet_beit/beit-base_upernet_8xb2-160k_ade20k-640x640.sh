#!/usr/bin/env bash
#SBATCH --time=17:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=corruption_beit-base_upernet_8xb2-160k_ade20k-640x640
#SBATCH --output=slurm/corruption_beit-base_upernet_8xb2-160k_ade20k-640x640.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310
cd mmsegmentation

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
)

for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/ade20k/upernet/beit-base_upernet_8xb2-160k_ade20k-640x640.py \
        ../checkpoint_files/ade20k/upernet/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/ade20k/upernet/beit-base_upernet_8xb2-160k_ade20k-640x640/$corruption
    echo "Finished processing corruption type: $corruption"
done

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime"
