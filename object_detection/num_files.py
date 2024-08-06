import os

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)

print("Num of files to train (r50, r101, swin, convnext):", len(filenames_to_train))
print("Num of files to test (r50, r101, swin, convnext):", len(filenames_to_test))


def num_convnext_files():
    num_convnext_files = 0
    num_voc_convnext_files = 0
    num_coco_convnext_files = 0
    for filename in filenames_to_train:
        if "convnext" in filename:
            num_convnext_files += 1
            if "voc" in filename:
                num_voc_convnext_files += 1
            elif "coco" in filename:
                num_coco_convnext_files += 1
    print("Num of voc convnext files to train:", num_voc_convnext_files)
    print("Num of coco convnext files to train:", num_coco_convnext_files)
    print("Num of convnext files to train:", num_convnext_files)


def num_swin_files():
    num_swin_files = 0
    num_voc_swin_files = 0
    num_coco_swin_files = 0
    for filename in filenames_to_train:
        if "swin" in filename:
            num_swin_files += 1
            if "voc" in filename:
                num_voc_swin_files += 1
            elif "coco" in filename:
                num_coco_swin_files += 1
    print("Num of voc swin files to train:", num_voc_swin_files)
    print("Num of coco swin files to train:", num_coco_swin_files)
    print("Num of swin files to train:", num_swin_files)


num_convnext_files()
num_swin_files()

from mmengine.config import Config

# Load the configuration file
swin_cfg = Config.fromfile("configs_to_train/glip_swin-b_coco.py")

# Print the current load_from attribute
print("Before deletion:", swin_cfg.load_from)

# Delete the load_from attribute
swin_cfg.pop("load_from", "Not found")
# del swin_cfg.load_from
# del swin_cfg["load_from"]

# Verify that the load_from attribute has been removed
print("After deletion:", hasattr(swin_cfg, "load_from"))

print(swin_cfg.load_from)
