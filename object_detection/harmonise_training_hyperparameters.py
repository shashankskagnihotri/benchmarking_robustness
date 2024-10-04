import os
from mmengine.config import Config
import re


def file_mover(current_file_path, path_destination_folder):
    os.rename(
        current_file_path,
        f"{path_destination_folder}/{os.path.basename(current_file_path)}",
    )


def index_of_custom_hook(cfg, hook_type):
    if hasattr(cfg, "custom_hooks") and cfg.custom_hooks:
        for i, hook in enumerate(cfg.custom_hooks):
            if hook.type == hook_type:
                return i
    return None


def extract_head_backbone_dataset(filename):
    """
    Extracts the head, backbone, and dataset name from a given filename.

    Parameters:
    filename (str): The filename from which to extract the head, backbone, and dataset.

    Returns:
    tuple: A tuple containing the head, backbone, and dataset as strings.
    """
    pattern = r"^.*/(.+)_([^_]+)_([^_]+)\.py$"
    match = re.match(pattern, filename)
    if match:
        head = match.group(1)
        backbone = match.group(2)
        dataset = match.group(3)
        return head, backbone, dataset
    else:
        return None, None, None


def harmonise_training_hyperparameters(file4head, folder_path, path4backbonefile):
    print(f"Processing file: {file4head} from {folder_path}")
    cfg_head = Config.fromfile(f"{folder_path}/{file4head}")
    cfg_backbone = Config.fromfile(path4backbonefile)

    cfg_head.optim_wrapper = cfg_backbone.optim_wrapper
    cfg_head.optim_wrapper.type = "OptimWrapper"
    cfg_head.param_scheduler = cfg_backbone.param_scheduler

    cfg_head.max_epochs = cfg_backbone.max_epochs
    cfg_head.train_cfg.type = cfg_backbone.train_cfg.type
    cfg_head.train_cfg.max_epochs = cfg_backbone.train_cfg.max_epochs
    cfg_head.train_cfg.val_interval = cfg_backbone.train_cfg.val_interval

    if hasattr(cfg_head.train_cfg, "max_iters"):
        del cfg_head.train_cfg.max_iters

    # Handle EmmaHook
    idx_emma_head = index_of_custom_hook(cfg_head, "EmmaHook")
    idx_emma_back = index_of_custom_hook(cfg_backbone, "EmmaHook")
    if idx_emma_back is not None and idx_emma_head is not None:
        cfg_head.custom_hooks[idx_emma_head] = cfg_backbone.custom_hooks[idx_emma_back]
    elif idx_emma_back is None and idx_emma_head is not None:
        del cfg_head.custom_hooks[idx_emma_head]
    elif idx_emma_back is not None and idx_emma_head is None:
        if not hasattr(cfg_head, "custom_hooks") or cfg_head.custom_hooks is None:
            cfg_head.custom_hooks = []
        cfg_head.custom_hooks.append(cfg_backbone.custom_hooks[idx_emma_back])

    # Handle EarlyStoppingHook
    idx_early_head = index_of_custom_hook(cfg_head, "EarlyStoppingHook")

    # Define the early stopping hook based on the dataset
    if "coco" in file4head:
        early_stopping_hook = dict(
            type="EarlyStoppingHook",
            monitor="coco/bbox_mAP",
        )
    else:
        early_stopping_hook = dict(
            type="EarlyStoppingHook",
            monitor="pascal_voc/mAP",
        )

    # Remove existing EarlyStoppingHook if it exists
    if idx_early_head is not None:
        del cfg_head.custom_hooks[idx_early_head]
    # Ensure custom_hooks list exists
    if not hasattr(cfg_head, "custom_hooks") or cfg_head.custom_hooks is None:
        cfg_head.custom_hooks = []
    # Add the new EarlyStoppingHook
    cfg_head.custom_hooks.append(early_stopping_hook)

    # idx_emma_head = index_of_custom_hook(cfg_head, "EmmaHook")
    # idx_emma_back = index_of_custom_hook(cfg_backbone, "EmmaHook")
    # if idx_emma_back is not None and idx_emma_head is not None:
    #     cfg_head.custom_hooks[idx_emma_head] = cfg_backbone.custom_hooks[idx_emma_back]
    # elif idx_emma_back is None and idx_emma_head is not None:
    #     del cfg_head.custom_hooks[idx_emma_head]
    # elif idx_emma_back is not None and idx_emma_head is None:
    #     if not hasattr(cfg_head, "custom_hooks"):
    #         cfg_head.custom_hooks = []
    #     cfg_head.custom_hooks.append(cfg_backbone.custom_hooks[idx_emma_back])

    # idx_early_head = index_of_custom_hook(cfg_head, "EarlyStoppingHook")

    # coco_early_stopping_hook = dict(
    #     type="EarlyStoppingHook",
    #     monitor="coco/bbox_mAP",
    # )
    # voc_early_stopping_hook = dict(
    #     type="EarlyStoppingHook",
    #     monitor="pascal_voc/mAP",
    # )
    # early_stopping_hook = (
    #     coco_early_stopping_hook if "coco" in file4head else voc_early_stopping_hook
    # )

    # if idx_early_head:
    #     del cfg_head.custom_hooks[idx_early_head]
    # if not hasattr(cfg_head, "custom_hooks"):
    #     cfg_head.custom_hooks = []
    # cfg_head.custom_hooks.append(early_stopping_hook)

    cfg_head.default_hooks.param_scheduler = cfg_backbone.default_hooks.param_scheduler
    cfg_head.train_dataloader.batch_size = cfg_backbone.train_dataloader.batch_size

    if hasattr(cfg_head, "auto_scale_lr"):
        cfg_head.auto_scale_lr.enable = True
        cfg_head.auto_scale_lr.base_batch_size = 16
    else:
        cfg_head.auto_scale_lr = dict(enable=True, base_batch_size=16)

    # os.remove(f"{folder_path}/{file4head}")
    # print(f"Removed original file: {folder_path}/{file4head}")

    # head, _, dataset = extract_head_backbone_dataset(file4head)
    # _, backbone, _ = extract_head_backbone_dataset(path4backbonefile)

    output_path = f"./configs_to_train/{file4head}"

    cfg_head.dump(output_path)
    print(f"Saved modified config to: {output_path}")


paths4backbonefiles = [
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py",
    "./mmdetection/configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py",
    "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py",
    "./mmdetection/configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
]
folders_heads = [
    "./configs_to_train",
    "./configs_verified",
    "./configs_erroneous/verification",
    "./configs_to_test",
]
temp_folder = "./configs_temp"


for folder_heads in folders_heads:
    files4head = os.listdir(folder_heads)
    for file4head in files4head:
        if "swin" in file4head or "convnext" in file4head:
            file_mover(f"{folder_heads}/{file4head}", temp_folder)


for tempfile4head in os.listdir(temp_folder):
    for path4backbonefile in paths4backbonefiles:
        if (
            ("swin-b" in tempfile4head and "swin_b" in path4backbonefile)
            or ("swin-s" in tempfile4head and "swin-s" in path4backbonefile)
            or ("convnext-b" in tempfile4head and "convnext_b" in path4backbonefile)
            or ("convnext-s" in tempfile4head and "convnext-s" in path4backbonefile)
        ):
            print(
                f"Testing {tempfile4head} in folder {temp_folder} with {path4backbonefile}"
            )
            harmonise_training_hyperparameters(
                tempfile4head, temp_folder, path4backbonefile
            )

for tempfile4head in os.listdir(temp_folder):
    os.remove(f"{temp_folder}/{tempfile4head}")
    print(f"Removed temp file: {temp_folder}/{tempfile4head}")

# for path4backbonefile in paths4backbonefiles:
#         print(f"Testing {file4head} in folder {folder_heads} with {path4backbonefile}")
#         harmonise_training_hyperparameters(file4head, folder_heads, path4backbonefile)
