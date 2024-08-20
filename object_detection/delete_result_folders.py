import os
import shutil
from datetime import datetime

from tqdm import tqdm

# Define the path and date range
path_to_search = "./slurm/results"
date_from = datetime.strptime("2024-07-27", "%Y-%m-%d")
date_to = datetime.strptime("2024-07-28", "%Y-%m-%d")


# Helper function to get the creation time of a directory
def get_creation_time(path):
    return datetime.fromtimestamp(os.path.getmtime(path))


# Find second-level subdirectories and filter by creation date
directories_to_delete = []
for root, dirs, _ in os.walk(path_to_search):
    for directory in dirs:
        dir_path = os.path.join(root, directory)
        # Ensure we are only checking second-level subdirectories
        if len(os.path.relpath(dir_path, path_to_search).split(os.sep)) == 2:
            creation_time = get_creation_time(dir_path)
            if date_from <= creation_time < date_to:
                directories_to_delete.append(dir_path)

for directory in directories_to_delete:
    print(directory)

# Display and delete with progress bar
for directory in tqdm(directories_to_delete, desc="Deleting directories", unit="dir"):
    shutil.rmtree(directory)
