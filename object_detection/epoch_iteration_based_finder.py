import os
from mmengine.config import Config

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)

iteration_based_configs = []
epoch_based_configs = []


def epoch_iteration_based_finder(filename, folder_path):
    global epoch_based_counter
    global iteration_based_counter

    cfg = Config.fromfile(f"{folder_path}/{filename}")

    if cfg.train_cfg.type == "EpochBasedTrainLoop":
        epoch_based_configs.append(filename)
        epoch_based_counter += 1

    else:
        iteration_based_configs.append(filename)
        iteration_based_counter += 1


epoch_based_counter = 0
iteration_based_counter = 0

for filename in filenames_to_train:
    epoch_iteration_based_finder(filename, path_folder_to_train)


for filename in filenames_verified:
    epoch_iteration_based_finder(filename, path_folder_verified)

for filename in filenames_erroneous:
    epoch_iteration_based_finder(filename, path_folder_erroneous)

for filename in filenames_to_test:
    epoch_iteration_based_finder(filename, path_folder_to_test)


print(f"Number of epoch based configs: {epoch_based_counter}")
print(f"Epoch based configs: {epoch_based_configs}\n\n\n")
print(f"Number of iteration based configs: {iteration_based_counter}")
print(f"Iteration based configs: {iteration_based_configs}")
