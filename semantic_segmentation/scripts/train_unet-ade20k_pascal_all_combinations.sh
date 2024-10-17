#!/usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_unet-ade20k_pascal_all_combinations
#SBATCH --output=slurm/train_unet-ade20k_pascal_all_combinations_%a.out
#SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nico.sharei@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]] # mIoU: 1.3400
then
    python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-40k_ade20k-512x512_stride-340x340.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-40k_ade20k-512x512_stride-340x340
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]  # AssertionError: The input image size (512, 683) should be divisible by the whole downsample rate 16, when num_stages is 5, strides is (1, 1, 1, 1, 1), and downsamples is (True, True, True, True).
then
    python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-40k_ade20k-512x512_mode-whole.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-40k_ade20k-512x512_mode-whole
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]] # mIoU: 20.8900
then
    python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-40k_pascal_voc-512x512.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-40k_pascal_voc-512x512
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]] # mIoU: 21.5500
then
    python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-40k_pascal_voc-512x512_stride-340x340.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-40k_pascal_voc-512x512_stride-340x340
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]] # AssertionError: The input image size (512, 699) should be divisible by the whole downsample rate 16, when num_stages is 5, strides is (1, 1, 1, 1, 1), and downsamples is (True, True, True, True).

then
    python tools/train.py ../configs/unet/unet-s5-d16_fcn_4x4-40k_pascal_voc-512x512_mode-whole.py --work-dir ../work_dirs/unet-s5-d16_fcn_4x4-40k_pascal_voc-512x512_mode-whole
else
    echo "All submitted!"
fi


end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime

