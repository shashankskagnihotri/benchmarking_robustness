#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=06:59:59
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --job-name=starflow_sintel-clean_bim_pgd_cospgd_i20_inf
#SBATCH --output=slurm/starflow_sintel-clean_bim_pgd_cospgd_i20_inf_%A_%a.out
#SBATCH --error=slurm/starflow_sintel-clean_bim_pgd_cospgd_i20_inf_err_%A_%a.out
#SBATCH --array=0-2%3

model="starflow"
dataset="sintel-clean"
checkpoint="sintel"
targeteds="True False"
targets="negative zero"
norm="inf"
attacks="bim pgd cospgd"
iterations="20"
combinations=()

# Change to the appropriate directory
cd ../../../../

# Step 1: Create a list of all variable combinations
for targeted in $targeteds
do
    if [[ $targeted = "False" ]]
    then
        epsilons="4 8"
        alphas="0.01"          
    elif [[ $targeted = "True" ]]
    then
        epsilons="8"
        alphas="0.01"
    fi
    for epsilon in $epsilons
    do
        epsilon=$(echo "scale=10; $epsilon/255" | bc)
        for alpha in $alphas
        do
            for attack in $attacks
            do
                for iteration in $iterations
                do
                    if [[ $targeted = "True" ]]
                    then
                        for target in $targets
                        do
                            # Append each combination as a string to the combinations list
                            combinations+=("$model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $target")
                        done
                    else
                        # For non-targeted attacks
                        combinations+=("$model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted zero")
                    fi
                done
            done
        done
    done
done

# Step 2: Echo the content of the combinations array
echo "Generated combinations:"
for comb in "${combinations[@]}"; do
    echo "$comb"
done

# Step 3: Calculate the number of SLURM array tasks
total_jobs=${#combinations[@]}
echo "Total number of jobs: $total_jobs"
jobs_per_task=4  # Number of jobs to run per SLURM task

# Compute the total number of array tasks (rounding up)
array_length=$(( (total_jobs + jobs_per_task - 1) / jobs_per_task ))

# If running as part of a SLURM array job, determine start job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

start_job=$(( SLURM_ARRAY_TASK_ID * jobs_per_task ))

# Run up to 4 combinations on separate GPUs
for i in $(seq 0 3)  # Loop over 4 GPUs (0 to 3)
do
    jobnum=$(( start_job + i ))
    
    if [[ $jobnum -lt $total_jobs ]]
    then
        # Get the combination for the current task
        combination="${combinations[$jobnum]}"
        
        # Extract the individual variables from the combination string
        IFS=' ' read -r model checkpoint dataset attack iteration norm alpha epsilon targeted target <<< "$combination"
        
        # Set the GPU for this job
        export CUDA_VISIBLE_DEVICES=$i
        
        echo "Running job $model $checkpoint $dataset $attack $iteration $norm $alpha $epsilon $targeted $target on GPU $i (job $jobnum)"
        
        # Run the Python script with the corresponding parameters in the background (&)
        python attacks.py \
            $model \
            --pretrained_ckpt $checkpoint \
            --val_dataset $dataset \
            --attack $attack \
            --attack_iterations $iteration \
            --attack_norm $norm \
            --attack_alpha $alpha \
            --attack_epsilon $epsilon \
            --attack_targeted $targeted \
            --attack_target $target \
            --write_outputs &
    fi
done

# Wait for all background jobs to complete before exiting
wait
