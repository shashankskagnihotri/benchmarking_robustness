import os
from mmengine.config import Config


swin_b_cfg = Config.fromfile(
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py"
)
convnext_b_cfg = Config.fromfile(
    "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py"
)

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)

print("Filenames to train:", filenames_to_train)
print("Filenames verified:", filenames_verified)
print("Filenames erroneous:", filenames_erroneous)
print("Filenames to test:", filenames_to_test)


def change_training_implementation(filename, folder_path, backbone_cfg):
    print(f"Processing file: {filename} from {folder_path}")
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    cfg.optim_wrapper = backbone_cfg.optim_wrapper
    cfg.param_scheduler = backbone_cfg.param_scheduler
    cfg.max_epochs = backbone_cfg.max_epochs
    cfg.train_cfg.max_epochs = backbone_cfg.train_cfg.max_epochs

    if hasattr(cfg, "custom_hooks") and cfg.custom_hooks:
        num_hooks = len(cfg.custom_hooks)
        found_expema = False
        for h in range(num_hooks):
            if cfg.custom_hooks[h].type == "EMAHook":
                cfg.custom_hooks[h] = backbone_cfg.custom_hooks[0]
                found_expema = True
                break
        if not found_expema:
            cfg.custom_hooks.append(backbone_cfg.custom_hooks[0])
    else:
        cfg.custom_hooks = [backbone_cfg.custom_hooks[0]]

    cfg.default_hooks.param_scheduler = backbone_cfg.default_hooks.param_scheduler
    cfg.train_dataloader.batch_size = backbone_cfg.train_dataloader.batch_size

    if hasattr(cfg, "auto_scale_lr"):
        cfg.auto_scale_lr.enable = True
        cfg.auto_scale_lr.base_batch_size = 16
    else:
        cfg.auto_scale_lr = dict(enable=True, base_batch_size=16)

    os.remove(f"{folder_path}/{filename}")
    print(f"Removed original file: {folder_path}/{filename}")

    output_path = f"{path_folder_to_train}/{filename}"
    cfg.dump(output_path)
    print(f"Saved modified config to: {output_path}")


#! actualy only have to look through train and test. Remove and than safe into the same folder
for filename in filenames_to_train:
    if "swin-b" in filename:
        change_training_implementation(filename, path_folder_to_train, swin_b_cfg)
    elif "convnext-b" in filename:
        change_training_implementation(filename, path_folder_to_train, convnext_b_cfg)

for filename in filenames_verified:
    if "swin-b" in filename:
        change_training_implementation(filename, path_folder_verified, swin_b_cfg)
    elif "convnext-b" in filename:
        change_training_implementation(filename, path_folder_verified, convnext_b_cfg)

for filename in filenames_erroneous:
    if "swin-b" in filename:
        change_training_implementation(filename, path_folder_erroneous, swin_b_cfg)
    elif "convnext-b" in filename:
        change_training_implementation(filename, path_folder_erroneous, convnext_b_cfg)

for filename in filenames_to_test:
    if "swin-b" in filename:
        change_training_implementation(filename, path_folder_to_test, swin_b_cfg)
    elif "convnext-b" in filename:
        change_training_implementation(filename, path_folder_to_test, convnext_b_cfg)
