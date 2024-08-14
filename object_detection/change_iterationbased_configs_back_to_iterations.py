import os
from mmengine.config import Config
from mmengine.runner import Runner


path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)


def namefinder(filename):
    def neck(filename):
        return filename.split("_")[0]

    def backbone(filename):
        if "swin-b" in filename:
            return "swin-b"
        elif "convnext-b" in filename:
            return "convnext-b"
        elif "r50" in filename:
            return "r50"
        elif "r101" in filename:
            return "r101"
        else:
            return "unknown-backbone"

    def dataset(filename):
        if "coco" in filename:
            return "coco"
        elif "voc" in filename:
            return "voc0712"
        else:
            return "unknown-dataset"

    return neck(filename), backbone(filename), dataset(filename)


def calculate_iterations(epochs, batch_size, dataset_size):
    steps_per_epoch = dataset_size / batch_size
    total_iterations = epochs * steps_per_epoch

    return int(total_iterations)


def change_interations(filename, folder_path):
    cfg = Config.fromfile(f"{folder_path}/{filename}")
    cfg.work_dir = os.path.join(folder_path, "work_dir")

    runner = Runner.from_cfg(cfg)
    print(f"Filename: {filename}")

    # print(f"train_dataloader: {runner.train_dataloader}")

    # print(f"train_dataloader.dataset: {runner.train_dataloader.dataset}")

    print(f"len(train_dataloader.dataset): {len(runner.train_dataloader.dataset)}")
    print(f"len(val_dataloader.dataset): {len(runner.val_dataloader.dataset)}")
    print(f"len(test_dataloader.dataset): {len(runner.test_dataloader.dataset)}")

    num_train_images = len(runner.train_dataloader.dataset)

    cfg.train_cfg.type = "IterBasedTrainLoop"
    cfg.train_cfg.max_iters = calculate_iterations(
        epochs=cfg.max_epochs,
        batch_size=cfg.train_dataloader.batch_size,
        dataset_size=num_train_images,
    )
    if hasattr(cfg.train_cfg, "max_epochs"):
        cfg.train_cfg.pop("max_epochs", "Not found")

    cfg.train_cfg.val_interval = calculate_iterations(
        epochs=cfg.train_cfg.val_interval,
        batch_size=cfg.train_dataloader.batch_size,
        dataset_size=num_train_images,
    )

    # if hasattr(cfg.train_cfg, "max_epochs"):
    #     del cfg.train_cfg.max_epochs

    cfg.dump(f"{folder_path}/{filename}")


for filename in filenames_to_train:
    filepath = os.path.join(path_folder_to_train, filename)
    neck, backbone, dataset = namefinder(filename)
    if neck == "DiffusionDet" and backbone == "swin-b":
        change_interations(filename, path_folder_to_train)
    elif neck == "DiffusionDet" and backbone == "convnext-b":
        change_interations(filename, path_folder_to_train)

    # #! detic has batchsize [4,16] in train_dataloader find out how this works and what to assign
    # elif neck == "Detic" and backbone == "swin-b":
    #     change_interations(filename, path_folder_to_train)
    # elif neck == "Detic" and backbone == "convnext-b":
    #     change_interations(filename, path_folder_to_train)
