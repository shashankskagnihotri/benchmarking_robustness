#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --array=0-2%2
#SBATCH --job-name=hd3_kitti-2015_fgsm_two
#SBATCH --output=slurm/hd3_kitti-2015_fgsm_two_%A_%a.out
#SBATCH --error=slurm/hd3_kitti-2015_fgsm_two_err_%A_%a.out

model="hd3"
dataset="kitti-2015"
checkpoint="kitti"
targeteds="True False"
targets="negative zero"
norm="two"
attacks="fgsm"
jobnum=0
#SLURM_ARRAY_TASK_ID=0

cd ../../../../


for targeted in $targeteds
do
    epsilons="12.75"
    alphas="0.0001"
    for epsilon in $epsilons
    do
        epsilon=$(echo "scale=10; $epsilon/255" | bc)
        for alpha in $alphas
        do
            for attack in $attacks
            do

                if [[ $targeted = "True" ]]
                then
                    for target in $targets
                    do      
                        if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
                        then
                            echo "Running job $model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $target $jobnum"
                            python attacks.py \
                                $model \
                                --pretrained_ckpt $checkpoint \
                                --val_dataset $dataset \
                                --attack $attack \
                                --attack_norm $norm \
                                --attack_alpha $alpha \
                                --attack_epsilon $epsilon \
                                --attack_targeted $targeted \
                                --attack_target $target \
                                --write_outputs                                       
                            #SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                        fi
                        jobnum=$((jobnum + 1))
                    done
                else                           
                    if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
                    then
                        echo "Running job $model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $target $jobnum"
                        python attacks.py \
                            $model \
                            --pretrained_ckpt $checkpoint \
                            --val_dataset $dataset \
                            --attack $attack \
                            --attack_norm $norm \
                            --attack_alpha $alpha \
                            --attack_epsilon $epsilon \
                            --attack_targeted $targeted \
                            --attack_target "zero" \
                            --write_outputs 
                        #SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))
                    fi
                    jobnum=$((jobnum + 1))
                fi
            done
        done
    done
done