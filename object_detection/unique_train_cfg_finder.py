import os
from mmengine.config import Config

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"


def get_filenames(folder_paths):
    filenames = []
    for path in folder_paths:
        if os.path.exists(path):
            filenames.extend([(path, filename) for filename in os.listdir(path)])
    return filenames


def unique_train_cfg_finder(folder_paths):
    filenames = get_filenames(folder_paths)
    unique_train_cfgs = set()

    for folder_path, filename in filenames:
        try:
            cfg = Config.fromfile(f"{folder_path}/{filename}")
            train_cfg = cfg.get("train_cfg")
            if train_cfg is not None:
                unique_train_cfgs.add(str(train_cfg))
        except Exception as e:
            print(f"Error processing file {filename} in folder {folder_path}: {e}")

    return unique_train_cfgs


folder_paths = [
    path_folder_to_train,
    path_folder_verified,
    path_folder_erroneous,
    path_folder_to_test,
]

unique_train_cfgs = unique_train_cfg_finder(folder_paths)

for cfg in unique_train_cfgs:
    print(cfg)


# {'max_epochs': 24, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 150, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 4, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 12, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 25, 'type': 'EpochBasedTrainLoop', 'val_interval': 5}
# {'max_iters': 450000, 'type': 'IterBasedTrainLoop', 'val_interval': 75000}
# {'max_epochs': 20, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'dynamic_intervals': [(90, 1)], 'max_epochs': 33, 'type': 'EpochBasedTrainLoop', 'val_interval': 10}
# {'max_epochs': 8, 'type': 'EpochBasedTrainLoop', 'val_interval': 5}
# {'_scope_': 'mmdet', 'max_epochs': 300, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 6, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 36, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'dynamic_intervals': [(90, 1)], 'max_epochs': 100, 'type': 'EpochBasedTrainLoop', 'val_interval': 10}
# {'max_epochs': 100, 'type': 'EpochBasedTrainLoop', 'val_interval': 10}
# {'max_iters': 180000, 'type': 'IterBasedTrainLoop', 'val_interval': 180000}
# {'max_epochs': 91, 'type': 'EpochBasedTrainLoop', 'val_interval': 7}
# {'_scope_': 'mmdet', 'max_epochs': 100, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 25, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 273, 'type': 'EpochBasedTrainLoop', 'val_interval': 7}
# {'max_epochs': 16, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 300, 'type': 'EpochBasedTrainLoop', 'val_interval': 10}
# {'max_epochs': 50, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
# {'max_epochs': 8, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}
