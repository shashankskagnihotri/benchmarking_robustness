#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --job-name=snow_maskflownet
#SBATCH --output=slurm/snow/maskflownet/snow_maskflownet_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --error=slurm/snow/maskflownet/snow_maskflownet_%j.err

echo "Job started at $(date)"
start_time=$(date +%s)

MODEL="maskflownet"
PRETRAINED_CKPT="sintel"
VAL_DATASET="sintel-tr115-final"
ATTACK="weather"
DATASET_STAGE="training"
DATASET="Sintel"

LR=0.00001
STEPS=750
ALPH_MOTION=1000
ALPH_MOTIONOFFSET=1000
LEARN_OFFSET=True
LEARN_MOTIONOFFSET=True
LEARN_TRANSPARENCY=True
LEARN_COLOR=True
DEPTH_CHECK=True
DO_MOTIONBLUR=False
WEATHER_DATA="/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/weather_sampledata/particles_3000_npz/weather_snow_3000"
UNREGISTERED_ARTIFACTS=True
NO_FLAKE_DAT=False
RENDERING_METHOD=additive
TRANSPARENCY_SCALE=0.75
FLAKESIZE_MAX=71
DEPTH_DECAY=9
MOTION_Y=0.2
FLAKE_R=255
FLAKE_G=255
FLAKE_B=255
WEATHER_FRAME_PER_SCENE=5

ATTACKS_SCRIPT_PATH="/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/attacks.py"

ATTACK_TARGETED="True False"
TARGETS="negative zero"

jobnum=0

for TARGET in $ATTACK_TARGETED
do
    for TARGET_CONDITION in $TARGETS
    do

        if [[ $SLURM_ARRAY_TASK_ID -eq $jobnum ]]
        then
            OUTPUT_PATH="outputs/validate/snow_tr115/targeted_${TARGET}/condition_${TARGET_CONDITION}"

            echo "Running job $MODEL $PRETRAINED_CKPT $VAL_DATASET $ATTACK $STEPS $LEARN_OFFSET $LEARN_MOTIONOFFSET $LEARN_COLOR $LEARN_TRANSPARENCY $ALPH_MOTION $ALPH_MOTIONOFFSET $TARGET $TARGET_CONDITION $jobnum"

            python $ATTACKS_SCRIPT_PATH $MODEL \
                --pretrained_ckpt $PRETRAINED_CKPT \
                --val_dataset $VAL_DATASET \
                --attack $ATTACK \
                --weather_dataset_stage $DATASET_STAGE \
                --weather_dataset $DATASET \
                --weather_optimizer Adam \
                --weather_lr $LR \
                --weather_steps $STEPS \
                --weather_alph_motion $ALPH_MOTION \
                --weather_alph_motionoffset $ALPH_MOTIONOFFSET \
                --weather_learn_offset $LEARN_OFFSET \
                --weather_learn_motionoffset $LEARN_MOTIONOFFSET \
                --weather_learn_transparency $LEARN_TRANSPARENCY \
                --weather_learn_color $LEARN_COLOR \
                --weather_depth_check $DEPTH_CHECK \
                --weather_do_motionblur $DO_MOTIONBLUR \
                --weather_data $WEATHER_DATA \
                --weather_unregistered_artifacts $UNREGISTERED_ARTIFACTS \
                --weather_no_flake_dat $NO_FLAKE_DAT \
                --weather_rendering_method $RENDERING_METHOD \
                --weather_transparency_scale $TRANSPARENCY_SCALE \
                --weather_flakesize_max $FLAKESIZE_MAX \
                --weather_depth_decay $DEPTH_DECAY \
                --weather_motion_y $MOTION_Y \
                --weather_flake_r $FLAKE_R \
                --weather_flake_g $FLAKE_G \
                --weather_flake_b $FLAKE_B \
                --weather_frame_per_scene $WEATHER_FRAME_PER_SCENE \
                --attack_targeted $TARGET \
                --attack_target $TARGET_CONDITION \
                --output_path $OUTPUT_PATH
        fi
        jobnum=$((jobnum + 1))
    done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "Job completed at $(date)"
echo "Runtime: $runtime seconds"
