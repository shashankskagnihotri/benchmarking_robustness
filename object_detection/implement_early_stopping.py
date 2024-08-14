import os
from mmengine.config import Config

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"


#! is tested and works good. Now find out setting of early stopping


def find_early_stopping_implementations(filename, folder_path):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    if hasattr(cfg, "custom_hooks") and cfg.custom_hooks:
        num_hooks = len(cfg.custom_hooks)
        found_early_stopping = False
        for h in range(num_hooks):
            if cfg.custom_hooks[h].type == "EarlyStoppingHook":
                if "coco" in filename:
                    cfg.custom_hooks[h] = (
                        dict(
                            type="EarlyStoppingHook",
                            monitor="coco/bbox_mAP",
                        ),
                    )
                elif "voc" in filename:
                    cfg.custom_hooks[h] = (
                        dict(
                            type="EarlyStoppingHook",
                            monitor="pascal_voc/mAP",
                        ),
                    )
                found_early_stopping = True
                break
        if not found_early_stopping:
            if "coco" in filename:
                cfg.custom_hooks.append(
                    dict(
                        type="EarlyStoppingHook",
                        monitor="coco/bbox_mAP",
                    ),
                )
            elif "voc" in filename:
                cfg.custom_hooks.append(
                    dict(
                        type="EarlyStoppingHook",
                        monitor="pascal_voc/mAP",
                    ),
                )
    else:
        if "coco" in filename:
            cfg.custom_hooks = [
                dict(
                    type="EarlyStoppingHook",
                    monitor="coco/bbox_mAP",
                ),
            ]
        elif "voc" in filename:
            cfg.custom_hooks = [
                dict(
                    type="EarlyStoppingHook",
                    monitor="pascal_voc/mAP",
                ),
            ]
    cfg.dump(f"{folder_path}/{filename}")


test_file = "cascade_rcnn_r50_voc0712.py"
test_folder_path = "cfg_experiments"

find_early_stopping_implementations(test_file, test_folder_path)
