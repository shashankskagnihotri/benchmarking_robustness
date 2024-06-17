import os
from mmengine.config import Config
from config_maker import which

#! co-dino, yolo etc.

path_folder_to_train = "./configs_to_train"
path_folder_erroneous = "./configs_erroneous/verification"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_erroneous = os.listdir(path_folder_erroneous)

#! do for training and erroneous
for filename in filenames_to_train:
    filepath = os.join(path_folder_to_train, filename)
    neck, backbone, dataset = which(filepath)

    if neck == "codino" and dataset == "coco":
        pass
    elif neck == "codino" and dataset == "voc":
        pass
    elif neck == "yolo" and backbone == "swin-b" and ..:
        cfg = Config.fromfile(filepath)
        cfg.model.neck.in_channels = 
        cfg.model.backbone
    elif neck == "yolo" and ...:
        pass

