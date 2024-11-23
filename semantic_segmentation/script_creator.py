import os

def generate_sh_scripts(directory_path):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        main_path = os.path.basename(os.path.normpath(directory_path))
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)
            file_name = str(file)  # Current file name
            parent_folder_name = os.path.basename(os.path.dirname(file_path))
            
            # Create a .sh file for the current file
            sh_file_name = f"{file_name[:-3]}.sh"
            output_sh_file = os.path.join(current_dir, sh_file_name)
            
            # Open the .sh script file in write mode
            with open(output_sh_file, 'w') as sh_file:
                # Write the shebang line and SLURM directives
                sh_file.write("#!/usr/bin/env bash\n")
                sh_file.write("#SBATCH --time=15:00:00\n")
                sh_file.write("#SBATCH --nodes=1\n")
                sh_file.write("#SBATCH --ntasks=1\n")
                sh_file.write("#SBATCH --partition=gpu_4\n")
                sh_file.write("#SBATCH --gres=gpu:1\n")
                sh_file.write("#SBATCH --mem=100G\n")
                sh_file.write("#SBATCH --cpus-per-task=16\n")
                sh_file.write(f"#SBATCH --job-name=corruption_{file_name[:-3]}\n")
                sh_file.write(f"#SBATCH --output=slurm/corruption_{file_name[:-3]}.out\n")
                sh_file.write("#SBATCH --mail-type=ALL\n")
                sh_file.write("#SBATCH --mail-user=mehmet.kacar@students.uni-mannheim.de\n\n")

                sh_file.write('echo "Started at $(date)";\n')
                sh_file.write('source activate py310\n')
                sh_file.write('cd mmsegmentation\n\n')

                # List of corruption methods
                sh_file.write("corruptions=(\n")
                corruptions = [
                    'gaussian_noise', 'shot_noise', 'impulse_noise',
                    'defocus_blur', 'glass_blur', 'motion_blur',
                    'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform',
                    'pixelate', 'jpeg_compression'
                ]
                for corruption in corruptions:
                    sh_file.write(f"    '{corruption}'\n")
                sh_file.write(")\n\n")

                # Loop through each corruption type and write the commands
                
                sh_file.write(f"for corruption in \"${{corruptions[@]}}\"; do\n")
                sh_file.write(f'    echo "Processing corruption type: $corruption"\n')
                sh_file.write(f"    python tools/test.py ../corruptions/corruption_config/{main_path}/{parent_folder_name}/{file_name} \\\n")
                sh_file.write(f"        ../checkpoint_files/{main_path}/{parent_folder_name}/manuel_entry \\\n")
                sh_file.write(f'        --cfg-options \"model.data_preprocessor.corruption=$corruption" \\\n')
                sh_file.write(f"        --work-dir ../corruptions/work_dirs/{main_path}/{parent_folder_name}/{file_name[:-3]}/$corruption\n")
                sh_file.write('    echo "Finished processing corruption type: $corruption"\n')
                sh_file.write('done\n\n')
                sh_file.write('end=$(date +%s)\n')
                sh_file.write('runtime=$((end-start))\n')
                sh_file.write('echo "Runtime: $runtime"\n')

            print(f"Bash script generated: {output_sh_file}")

# Example usage:
input_directory = "./corruptions/corruption_config/cityscapes"
generate_sh_scripts(input_directory)
