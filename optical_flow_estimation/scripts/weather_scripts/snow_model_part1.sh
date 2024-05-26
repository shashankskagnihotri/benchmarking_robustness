#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=45:00:00
#SBATCH --job-name=snow_g_ra
#SBATCH --output=slurm/snow_g_ra_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --error=slurm/snow_g_r_%j.err

echo "Job started at $(date)"
start_time=$(date +%s)

MODELS="gma raft"
PRETRAINED_CKPT="sintel"
VAL_DATASETS="sintel-clean-val sintel-final-val"
ATTACK="weather"
WEATHER_STEPS=1
WEATHER_MOTIONBLUR_SAMPLES=10
WEATHER_LEARN_OFFSET=True
WEATHER_LEARN_MOTIONOFFSET=True
WEATHER_LEARN_COLOR=True
WEATHER_LEARN_TRANSPARENCY=True
WEATHER_ALPH_MOTION=1000.
WEATHER_ALPH_MOTIONOFFSET=1000.
WEATHER_DATA="/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/weather_sampledata/particles_3000_npz/weather_snow_3000"
WEATHER_DATASET="Sintel"
WEATHER_DATASET_STAGE="training"
WEATHER_DEPTH_CHECK=True
WEATHER_MODEL_ITERS=3

ATTACKS_SCRIPT_PATH="/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/attacks.py"

ATTACK_TARGETED="True False"
TARGETS="negative zero"

jobnum=0

for MODEL in $MODELS
do
    for TARGET in $ATTACK_TARGETED
    do
        for TARGET_CONDITION in $TARGETS
        do
            for VAL_DATASET in $VAL_DATASETS
            do
                if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
                then
                    if [[ $VAL_DATASET == "sintel-clean-val" ]]; then
                        OUTPUT_PATH="outputs/validate/snow/clean"
                    elif [[ $VAL_DATASET == "sintel-final-val" ]]; then
                        OUTPUT_PATH="outputs/validate/snow/final"
                    fi

                    OUTPUT_PATH="${OUTPUT_PATH}/${MODEL}/targeted_${TARGET}/condition_${TARGET_CONDITION}"

                    echo "Running job $MODEL $PRETRAINED_CKPT $VAL_DATASET $ATTACK $WEATHER_STEPS $WEATHER_MOTIONBLUR_SAMPLES $WEATHER_LEARN_OFFSET $WEATHER_LEARN_MOTIONOFFSET $WEATHER_LEARN_COLOR $WEATHER_LEARN_TRANSPARENCY $WEATHER_ALPH_MOTION $WEATHER_ALPH_MOTIONOFFSET $TARGET $TARGET_CONDITION $jobnum"

                    python $ATTACKS_SCRIPT_PATH $MODEL \
                        --pretrained_ckpt $PRETRAINED_CKPT \
                        --val_dataset $VAL_DATASET \
                        --attack $ATTACK \
                        --weather_steps $WEATHER_STEPS \
                        --weather_motionblur_samples $WEATHER_MOTIONBLUR_SAMPLES \
                        --weather_learn_offset $WEATHER_LEARN_OFFSET \
                        --weather_learn_motionoffset $WEATHER_LEARN_MOTIONOFFSET \
                        --weather_learn_color $WEATHER_LEARN_COLOR \
                        --weather_learn_transparency $WEATHER_LEARN_TRANSPARENCY \
                        --weather_alph_motion $WEATHER_ALPH_MOTION \
                        --weather_alph_motionoffset $WEATHER_ALPH_MOTIONOFFSET \
                        --weather_data $WEATHER_DATA \
                        --weather_dataset $WEATHER_DATASET \
                        --weather_dataset_stage $WEATHER_DATASET_STAGE \
                        --attack_targeted $TARGET \
                        --attack_target $TARGET_CONDITION \
                        --weather_depth_check $WEATHER_DEPTH_CHECK \
                        --weather_model_iters $WEATHER_MODEL_ITERS \
                        --output_path $OUTPUT_PATH 
                        
                fi
                jobnum=$((jobnum + 1))
            done
        done
    done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "Job completed at $(date)"
echo "Runtime: $runtime seconds"
