import os
from mmengine.config import Config
from rtmdet_swin_b_coco import (
    train_dataloader as train_dataloader_rtmdet_swin_b_coco,
    train_pipeline as train_pipeline_rtmdet_swin_b_coco,
    train_pipeline_stage2 as train_pipeline_stage2_rtmdet_swin_b_coco,
)
from rtmdet_convnext_b_coco import (
    train_dataloader as train_dataloader_rtmdet_convnext_b_coco,
    train_pipeline as train_pipeline_rtmdet_convnext_b_coco,
    train_pipeline_stage2 as train_pipeline_stage2_rtmdet_convnext_b_coco,
)

# Load reference configs
swin_b_cfg = Config.fromfile("rtmdet_swin_b_coco.py")
convnext_b_cfg = Config.fromfile("rtmdet_convnext_b_coco.py")

dataset_rtmdet_swin_b_coco = swin_b_cfg.train_dataloader.dataset
dataset_rtmdet_convnext_b_coco = convnext_b_cfg.train_dataloader.dataset

if train_pipeline_rtmdet_swin_b_coco == train_pipeline_stage2_rtmdet_swin_b_coco:
    print("swin_b stage1 == stage2")
else:
    print("swin_b stage1 != stage2")

if (
    train_pipeline_rtmdet_convnext_b_coco
    == train_pipeline_stage2_rtmdet_convnext_b_coco
):
    print("convnext_b stage1 == stage2")
else:
    print("convnext_b stage1 != stage2")

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)

print(f"Number of configs to train: {len(filenames_to_train)}")
print(f"Number of verified, correct configs: {len(filenames_verified)}")
print(f"Number of erroneous configs: {len(filenames_erroneous)}")
print(f"Number of configs to test: {len(filenames_to_test)}")

counter_coco_files = 0

counter_all_equal_train_dataloader_swin_b = 0
counter_same_dataset_swin_b = 0
counter_first_same_dataset_pipeline_swin_b = 0
counter_second_same_dataset_pipeline_swin_b = 0

counter_all_equal_train_dataloader_convnext_b = 0
counter_same_dataset_convnext_b = 0
counter_first_same_dataset_pipeline_convnext_b = 0
counter_second_same_dataset_pipeline_convnext_b = 0

# Keep track of filenames for matches
matched_configs_train_dataloader_swin_b = []
matched_configs_dataset_swin_b = []
matched_configs_first_pipeline_swin_b = []
matched_configs_second_pipeline_swin_b = []

matched_configs_train_dataloader_convnext_b = []
matched_configs_dataset_convnext_b = []
matched_configs_first_pipeline_convnext_b = []
matched_configs_second_pipeline_convnext_b = []


def check_config_file(filename, folder_path):
    global counter_coco_files
    global counter_all_equal_train_dataloader_swin_b
    global counter_same_dataset_swin_b
    global counter_first_same_dataset_pipeline_swin_b
    global counter_second_same_dataset_pipeline_swin_b
    global counter_all_equal_train_dataloader_convnext_b
    global counter_same_dataset_convnext_b
    global counter_first_same_dataset_pipeline_convnext_b
    global counter_second_same_dataset_pipeline_convnext_b

    if "coco" in filename:
        counter_coco_files += 1
        cfg = Config.fromfile(f"{folder_path}/{filename}")

        # Check against rtmdet_swin_b_coco
        if cfg.train_dataloader == train_dataloader_rtmdet_swin_b_coco:
            counter_all_equal_train_dataloader_swin_b += 1
            matched_configs_train_dataloader_swin_b.append(filename)
        if cfg.train_dataloader.dataset == dataset_rtmdet_swin_b_coco:
            counter_same_dataset_swin_b += 1
            matched_configs_dataset_swin_b.append(filename)
        if cfg.train_pipeline == train_pipeline_rtmdet_swin_b_coco:
            counter_first_same_dataset_pipeline_swin_b += 1
            matched_configs_first_pipeline_swin_b.append(filename)
        if (
            hasattr(cfg, "train_pipeline_stage2")
            and cfg.train_pipeline_stage2 == train_pipeline_stage2_rtmdet_swin_b_coco
        ):
            counter_second_same_dataset_pipeline_swin_b += 1
            matched_configs_second_pipeline_swin_b.append(filename)

        # Check against rtmdet_convnext_b_coco
        if cfg.train_dataloader == train_dataloader_rtmdet_convnext_b_coco:
            counter_all_equal_train_dataloader_convnext_b += 1
            matched_configs_train_dataloader_convnext_b.append(filename)
        if cfg.train_dataloader.dataset == dataset_rtmdet_convnext_b_coco:
            counter_same_dataset_convnext_b += 1
            matched_configs_dataset_convnext_b.append(filename)
        if cfg.train_pipeline == train_pipeline_rtmdet_convnext_b_coco:
            counter_first_same_dataset_pipeline_convnext_b += 1
            matched_configs_first_pipeline_convnext_b.append(filename)
        if (
            hasattr(cfg, "train_pipeline_stage2")
            and cfg.train_pipeline_stage2
            == train_pipeline_stage2_rtmdet_convnext_b_coco
        ):
            counter_second_same_dataset_pipeline_convnext_b += 1
            matched_configs_second_pipeline_convnext_b.append(filename)


for filename in filenames_to_train:
    check_config_file(filename, path_folder_to_train)

for filename in filenames_verified:
    check_config_file(filename, path_folder_verified)

for filename in filenames_erroneous:
    check_config_file(filename, path_folder_erroneous)

for filename in filenames_to_test:
    check_config_file(filename, path_folder_to_test)

print(f"Number of configs with coco in filename: {counter_coco_files}")

print(
    f"Number of configs with same train_dataloader as rtmdet_swin_b_coco: {counter_all_equal_train_dataloader_swin_b}"
)
print(f"Matching configs: {matched_configs_train_dataloader_swin_b}")
print(
    f"Number of configs with same dataset as rtmdet_swin_b_coco: {counter_same_dataset_swin_b}"
)
print(f"Matching configs: {matched_configs_dataset_swin_b}")
print(
    f"Number of configs with same first pipeline as rtmdet_swin_b_coco: {counter_first_same_dataset_pipeline_swin_b}"
)
print(f"Matching configs: {matched_configs_first_pipeline_swin_b}")
print(
    f"Number of configs with same second pipeline as rtmdet_swin_b_coco: {counter_second_same_dataset_pipeline_swin_b}"
)
print(f"Matching configs: {matched_configs_second_pipeline_swin_b}")

print(
    f"Number of configs with same train_dataloader as rtmdet_convnext_b_coco: {counter_all_equal_train_dataloader_convnext_b}"
)
print(f"Matching configs: {matched_configs_train_dataloader_convnext_b}")
print(
    f"Number of configs with same dataset as rtmdet_convnext_b_coco: {counter_same_dataset_convnext_b}"
)
print(f"Matching configs: {matched_configs_dataset_convnext_b}")
print(
    f"Number of configs with same first pipeline as rtmdet_convnext_b_coco: {counter_first_same_dataset_pipeline_convnext_b}"
)
print(f"Matching configs: {matched_configs_first_pipeline_convnext_b}")
print(
    f"Number of configs with same second pipeline as rtmdet_convnext_b_coco: {counter_second_same_dataset_pipeline_convnext_b}"
)
print(f"Matching configs: {matched_configs_second_pipeline_convnext_b}")
