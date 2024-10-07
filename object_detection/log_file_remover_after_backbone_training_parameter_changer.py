import os


log_path = "slurm/work_dir"
files = os.listdir(log_path)


def remove_log_folder_of_file(file, log_path):
    specific_log_folder = f"{log_path}/{os.path.splitext(file)[0]}"
    print(f"Removing log folder: {specific_log_folder}")
    os.system(f"rm -r {specific_log_folder}")


for file in files:
    if "swin-b" in file:
        remove_log_folder_of_file(file, log_path)
    elif "convnext-b" in file:
        remove_log_folder_of_file(file, log_path)
