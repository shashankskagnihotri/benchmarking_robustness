#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=corruption_segformer_mit-b4_8xb1-160k_cityscapes-1024x1024
#SBATCH --output=slurm/corruption_segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310
cd mmsegmentation

corruptions=(
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
    python tools/test.py ../corruptions/corruption_config/cityscapes/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py \
        ../checkpoint_files/cityscapes/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/cityscapes/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024/$corruption
    echo "Finished processing corruption type: $corruption"
done

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime"
