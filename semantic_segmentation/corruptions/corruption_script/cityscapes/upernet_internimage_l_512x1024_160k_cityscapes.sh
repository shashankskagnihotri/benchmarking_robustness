#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=corruption_upernet_internimage_l_512x1024_160k_cityscapes
#SBATCH --output=slurm/corruption_upernet_internimage_l_512x1024_160k_cityscapes.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310
cd mmsegmentation

corruptions=(
    # 'glass_blur'
    # 'motion_blur'
    # 'zoom_blur'
    # 'snow'
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
    python tools/test.py ../corruptions/corruption_config/cityscapes/internimage/upernet_internimage_l_512x1024_160k_cityscapes.py \
        ../checkpoint_files/cityscapes/internimage/upernet_internimage_l_512x1024_160k_cityscapes.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/cityscapes/internimage/upernet_internimage_l_512x1024_160k_cityscapes/$corruption
    echo "Finished processing corruption type: $corruption"
done

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime"
