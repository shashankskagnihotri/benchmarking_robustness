import os
from mmengine.config import Config


def extract_coco_dataloader(folder_path, filename):
    # Initialize counters and dictionaries
    counters = {"train": 0, "val": 0, "test": 0}
    others = {"train": {}, "val": {}, "test": {}}
    excluded_files = []

    if "coco" in filename:
        cfg_path = os.path.join(folder_path, filename)
        cfg = Config.fromfile(cfg_path)

        reference_ann_files = {
            "train": "annotations/instances_train2017.json",
            "val": "annotations/instances_val2017.json",
            "test": "annotations/instances_val2017.json",  # Change if different reference needed
        }

        dataloaders = {
            "train": getattr(cfg, "train_dataloader", None),
            "val": getattr(cfg, "val_dataloader", None),
            "test": getattr(cfg, "test_dataloader", None),
        }

        for key, dataloader in dataloaders.items():
            if dataloader:
                ann_file = None

                # First check for ann_file in dataloader.dataset
                if hasattr(dataloader.dataset, "ann_file"):
                    ann_file = dataloader.dataset.ann_file
                # If not found, check for ann_file in dataloader.dataset.dataset
                elif hasattr(dataloader.dataset, "dataset") and hasattr(
                    dataloader.dataset.dataset, "ann_file"
                ):
                    ann_file = dataloader.dataset.dataset.ann_file

                if ann_file:
                    if ann_file == reference_ann_files[key]:
                        counters[key] += 1
                    else:
                        others[key][filename] = ann_file
                else:
                    excluded_files.append(filename)
            else:
                excluded_files.append(filename)

    return counters, others, excluded_files


def process_directory(path):
    filenames = os.listdir(path)
    total_counters = {"train": 0, "val": 0, "test": 0}
    total_others = {"train": {}, "val": {}, "test": {}}
    total_excluded_files = []

    for filename in filenames:
        counters, others, excluded_files = extract_coco_dataloader(path, filename)
        for key in total_counters:
            total_counters[key] += counters[key]
            total_others[key].update(others[key])
        total_excluded_files.extend(excluded_files)

    return total_counters, total_others, total_excluded_files


paths = [
    "./configs_to_train",
    "./configs_verified",
    "./configs_erroneous/verification",
    "./configs_to_test",
]

overall_counters = {"train": 0, "val": 0, "test": 0}
overall_others = {"train": {}, "val": {}, "test": {}}
overall_excluded_files = []

for path in paths:
    counters, others, excluded_files = process_directory(path)
    for key in overall_counters:
        overall_counters[key] += counters[key]
        overall_others[key].update(others[key])
    overall_excluded_files.extend(excluded_files)

print("Counter equal train ann files:", overall_counters["train"])
print("Counter equal val ann files:", overall_counters["val"])
print("Counter equal test ann files:", overall_counters["test"])

print("Other train ann files:", overall_others["train"])
print("Other val ann files:", overall_others["val"])
print("Other test ann files:", overall_others["test"])

print("Files excluded due to missing or different 'ann_file':", overall_excluded_files)
