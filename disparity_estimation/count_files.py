import os
import json

def get_files_with_size(directory):
    """Get a list of files with their sizes in the given directory."""
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            filesize = os.path.getsize(filepath)
            relative_path = os.path.relpath(filepath, directory)  # Relative path from the directory
            files_list.append((relative_path, filesize, filepath))
    return files_list

def compare_file_lists(list1, list2):
    """Compare two lists of files with their sizes and return the differences."""
    set1 = set(list1)
    set2 = set(list2)

    only_in_first = set1 - set2
    only_in_second = set2 - set1
    differences = list(only_in_first) + list(only_in_second)
    
    return differences

def print_differences(differences):
    """Print the differences in a readable format."""
    if not differences:
        print("The directories have identical files and sizes.")
    else:
        print("Differences found:")
        for diff in differences:
            print(f"File: {diff[0]}, Size: {diff[1]} bytes")


def write_list_to_file(file_list, output_file):
    """Write the list of files with sizes to a file."""
    with open(output_file, 'w') as f:
        json.dump(file_list, f, indent=4)

# Example usage:
# "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0" ===> 53520
# /pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/tmp

current_files = "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0/frames_finalpass"
#new_files = "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/tmp/frames_finalpass"

files_list1 = get_files_with_size(current_files)
#files_list2 = get_files_with_size(new_files)

# Save the file lists to disk
write_list_to_file(files_list1, 'files_list1.json')
#write_list_to_file(files_list2, 'files_list2.json')

# differences = compare_file_lists(files_list1, files_list2)
# print_differences(differences)

print(f"Length: files_list1: {len(files_list1)}")
#print(f"Length: files_list2: {len(files_list2)}")

# ### DIFF

# import filecmp
# import os

# def compare_directories(dir1, dir2):
#     # Compare the directories
#     dir_comp = filecmp.dircmp(dir1, dir2)
#     dir_comp.report()  # Generates a report

#     # Recursively compare subdirectories
#     for sub_dir in dir_comp.common_dirs:
#         compare_directories(os.path.join(dir1, sub_dir), os.path.join(dir2, sub_dir))

# # Paths to the directories
# current_files = "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0/frames_finalpass"
# new_files = "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/tmp/frames_finalpass"

# # Start comparison
# compare_directories(current_files, new_files)

