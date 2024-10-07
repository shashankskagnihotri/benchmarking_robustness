import os
from mmengine.config import Config

# Load reference configs
swin_b_cfg = Config.fromfile(
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py"
)
convnext_b_cfg = Config.fromfile(
    "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py"
)

# Define paths
path_folder_to_check = "./configs_to_train"

# Get filenames
filenames_to_check = os.listdir(path_folder_to_check)

print("Filenames to check:", filenames_to_check)


def check_unification(filename, folder_path, reference_cfg):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    is_unified = True
    mismatches = []

    # Check key parameters
    if cfg.optim_wrapper != reference_cfg.optim_wrapper:
        mismatches.append("optim_wrapper")
        is_unified = False
    if cfg.param_scheduler != reference_cfg.param_scheduler:
        mismatches.append("param_scheduler")
        is_unified = False
    if cfg.max_epochs != reference_cfg.max_epochs:
        mismatches.append("max_epochs")
        is_unified = False
    if cfg.train_cfg.max_epochs != reference_cfg.train_cfg.max_epochs:
        mismatches.append("train_cfg.max_epochs")
        is_unified = False
    if cfg.default_hooks.param_scheduler != reference_cfg.default_hooks.param_scheduler:
        mismatches.append("default_hooks.param_scheduler")
        is_unified = False
    if cfg.train_dataloader.batch_size != reference_cfg.train_dataloader.batch_size:
        mismatches.append("train_dataloader.batch_size")
        is_unified = False
    if hasattr(cfg, "auto_scale_lr"):
        if cfg.auto_scale_lr != reference_cfg.auto_scale_lr:
            mismatches.append("auto_scale_lr")
            is_unified = False
    else:
        if "auto_scale_lr" in reference_cfg:
            mismatches.append("missing auto_scale_lr")
            is_unified = False

    # Custom hooks check
    custom_hooks_match = any(
        hook.type == "EMAHook" and hook == reference_cfg.custom_hooks[0]
        for hook in cfg.custom_hooks
    )
    if not custom_hooks_match:
        mismatches.append("custom_hooks.EMAHook")
        is_unified = False

    if is_unified:
        print(f"{filename}: Unified")
    else:
        print(f"{filename}: Not unified. Mismatches: {', '.join(mismatches)}")


# Check files
for filename in filenames_to_check:
    if "swin-b" in filename:
        check_unification(filename, path_folder_to_check, swin_b_cfg)
    elif "convnext-b" in filename:
        check_unification(filename, path_folder_to_check, convnext_b_cfg)
