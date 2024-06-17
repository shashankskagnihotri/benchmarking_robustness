import os
from mmengine.config import Config

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

folders = [
    path_folder_to_train,
    path_folder_verified,
    path_folder_erroneous,
    path_folder_to_test,
]


def find_keys(d, key_set):
    for key, value in d.items():
        key_set.add(key)
        if isinstance(value, dict):
            find_keys(value, key_set)


def collect_all_keys(config, section_key="train_cfg"):
    key_set = set()
    if section_key in config:
        find_keys(config[section_key], key_set)
    return key_set


def extract_custom_hooks(config, hooks_key="custom_hooks"):
    custom_hooks = []
    if hooks_key in config:
        hooks = config[hooks_key]
        if isinstance(hooks, list):
            for hook in hooks:
                if isinstance(hook, dict):
                    custom_hooks.append(hook)
    return custom_hooks


def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    else:
        return obj


all_keys = set()
all_custom_hooks = set()

for folder in folders:
    for filename in os.listdir(folder):
        config_path = os.path.join(folder, filename)
        try:
            cfg = Config.fromfile(config_path)
            keys = collect_all_keys(cfg)
            all_keys.update(keys)

            custom_hooks = extract_custom_hooks(cfg)
            for hook in custom_hooks:
                all_custom_hooks.add(make_hashable(hook))
        except Exception as e:
            print(f"Error loading {config_path}: {e}")

unique_custom_hook_types = {
    hook_dict.get("type") for hook in all_custom_hooks for hook_dict in [dict(hook)]
}

print("Collected Keys:", all_keys)
print("Unique Custom Hook Types:", unique_custom_hook_types)


files_with_default_hooks = []
files_with_custom_hooks = []
files_with_scope = []


for folder in folders:
    for filename in os.listdir(folder):
        config_path = os.path.join(folder, filename)
        cfg = Config.fromfile(config_path)
        if hasattr(cfg.train_cfg, "_scope_"):
            print(f"_scope_ found in train_cfg of {filename}")
            files_with_scope.append(filename)
            print(cfg.train_cfg)
            print("\n\n")
        if hasattr(cfg, "custom_hooks"):
            print(f"custom_hooks found in {filename}")
            files_with_custom_hooks.append(filename)
            print(cfg.custom_hooks)
            print("\n\n")
        if hasattr(cfg, "default_hooks"):
            print(f"default_hooks found in {filename}")
            files_with_default_hooks.append(filename)
            print(cfg.default_hooks)
            print("\n\n")


files_with_scope.sort()
files_with_default_hooks.sort()
files_with_custom_hooks.sort()


print("Files with _scope_ in train_cfg:", files_with_scope)
print("Files with default_hooks:", files_with_default_hooks)
print("Files with custom_hooks:", files_with_custom_hooks)


# Collected Keys: {'type', '_scope_', 'val_interval', 'max_iters', 'max_epochs', 'dynamic_intervals'}

# Files with _scope_ in train_cfg: ['EfficientDet_convnext-b_coco.py', 'EfficientDet_convnext-b_voc0712.py', 'EfficientDet_r101_coco.py', 'EfficientDet_r101_voc0712.py', 'EfficientDet_r50_coco.py', 'EfficientDet_r50_voc0712.py', 'EfficientDet_swin-b_coco.py', 'EfficientDet_swin-b_voc0712.py']
# Files with custom_hooks: ['EfficientDet_convnext-b_coco.py', 'EfficientDet_convnext-b_voc0712.py', 'EfficientDet_r101_coco.py', 'EfficientDet_r101_voc0712.py', 'EfficientDet_r50_coco.py', 'EfficientDet_r50_voc0712.py', 'EfficientDet_swin-b_coco.py', 'EfficientDet_swin-b_voc0712.py', 'rtmdet_convnext-b_coco.py', 'rtmdet_convnext-b_voc0712.py', 'rtmdet_r101_coco.py', 'rtmdet_r101_voc0712.py', 'rtmdet_r50_coco.py', 'rtmdet_r50_voc0712.py', 'rtmdet_swin-b_coco.py', 'rtmdet_swin-b_voc0712.py', 'yolox_convnext-b_coco.py', 'yolox_convnext-b_voc0712.py', 'yolox_r101_coco.py', 'yolox_r101_voc0712.py', 'yolox_r50_coco.py', 'yolox_r50_voc0712.py', 'yolox_swin-b_coco.py', 'yolox_swin-b_voc0712.py']


#! scope seems to have default value -> registers
#! custom_hooks all except yolox can be overwritten by the backbone specific custom_hooks (only training relevant not augumentation) for yolox check how to treat epochs in custom_hooks and what they specifically do


#! https://mmdetection.readthedocs.io/en/3.x/_modules/mmdet/engine/hooks/sync_norm_hook.html <- SyncNormHook : think should be kept
#! https://mmdetection.readthedocs.io/zh-cn/v3.1.0/_modules/mmdet/engine/hooks/yolox_mode_switch_hook.html <- YOLOXModeSwitchHook : think should be kept


# ? check dynamic intervalls, max_iters then ask how to proceed and implement it in this way
