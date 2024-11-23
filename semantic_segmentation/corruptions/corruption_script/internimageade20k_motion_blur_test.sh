#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=corruption_deeplabv3_r18-d8_4xb2-80k_cityscapes-512x1024
#SBATCH --output=slurm/corruption_deeplabv3_r18-d8_4xb2-80k_cityscapes-512x1024.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";
source activate py310
cd mmsegmentation

corruptions=(
    'motion_blur'
)

for corruption in "${corruptions[@]}"; do
    echo "Processing corruption type: $corruption"
    python tools/test.py ../corruptions/corruption_config/cityscapes/deeplabv3/deeplabv3_r18-d8_4xb2-80k_cityscapes-512x1024.py \
        ../checkpoint_files/cityscapes/deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth \
        --cfg-options "model.data_preprocessor.corruption=$corruption" \
        --work-dir ../corruptions/work_dirs/cityscapes/deeplabv3/deeplabv3_r18-d8_4xb2-80k_cityscapes-512x1024/$corruption
    echo "Finished processing corruption type: $corruption"
done

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime"