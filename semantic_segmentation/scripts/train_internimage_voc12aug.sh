#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=internimage_voc12aug_train
#SBATCH --output=slurm/internimage_voc12aug_train_%a.out
#SBATCH --array=1-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de

echo "Started at $(date)";

source activate openmmlab

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    python tools/train.py ../configs/internimage/upernet_internimage_s_160k_voc12aug_512x512.py --work-dir ../work_dirs/upernet_internimage_s_160k_voc12aug_512x512
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python tools/train.py ../configs/internimage/upernet_internimage_b_160k_voc12aug_512x512.py --work-dir ../work_dirs/upernet_internimage_b_160k_voc12aug_512x512
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python tools/train.py ../configs/internimage/upernet_internimage_t_160k_voc12aug_512x512.py --work-dir ../work_dirs/upernet_internimage_t_160k_voc12aug_512x512
else
    echo "All submitted"
fi 

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime