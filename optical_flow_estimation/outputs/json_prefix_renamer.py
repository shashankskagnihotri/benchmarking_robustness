import os

def rename_json_files_in_subdirectories(base_directory, prefix):
    # Walk through all files and subdirectories within the base directory
    for dirpath, _, filenames in os.walk(base_directory):
        for filename in filenames:
            # Check if the file is a JSON file and starts with "metrics" or "iteration_metrics"
            if filename.endswith('.json') and (filename.startswith("metrics") or filename.startswith("iteration_metrics")):
                # Check if the file already has the prefix
                if not filename.startswith(f"{prefix}_"):
                    # Construct the full file path
                    old_file_path = os.path.join(dirpath, filename)
                    
                    # Create the new filename with the specified prefix
                    new_filename = f"{prefix}_{filename}"
                    new_file_path = os.path.join(dirpath, new_filename)
                    
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    
                    # Log the rename action
                    print(f"Renamed: {old_file_path} -> {new_file_path}")

# Define the directory to search in and the prefix
base_directory = './validate'
prefix = 'j'  # Example: this prefix will rename 'metrics_example.json' to 'l_metrics_example.json'

# Call the function to rename files
rename_json_files_in_subdirectories(base_directory, prefix)
