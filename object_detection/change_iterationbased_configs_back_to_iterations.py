import os
from mmengine.config import Config
from config_maker import which


path_folder_to_train = "./configs_to_train"
filenames_to_train = os.listdir(path_folder_to_train)


def calculate_iterations(epochs, batch_size, dataset):
    if dataset == "coco":
        dataset_size = 0  #! get the right absolute dataset size
    elif dataset == "voc":
        dataset_size = 0  #! get the right absolute dataset size

    #! traing, val and testsizes (look into annotations? -> first decide, which annotations to use because of the switch to vocmetrics
    #! I think we just use the tainingsize as the approach should be training centric
    steps_per_epoch = dataset_size / batch_size
    total_iterations = epochs * steps_per_epoch

    return int(total_iterations)


def change_interations(filename, folder_path, dataset):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    # Trainingrelated
    #! paramscheduler is actually already converting into iterationbased

    cfg.train_cfg.type = "IterBasedTrainLoop"
    cfg.train_cfg.max_iters = calculate_iterations(
        epochs=cfg.max_epochs,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )

    cfg.default_hooks.checkpoint.by_epoch = False
    cfg.default_hooks.checkpoint.interval = calculate_iterations(
        epochs=cfg.default_hooks.checkpoint.interval,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )
    cfg.default_hooks.logger.interval = calculate_iterations(
        epochs=cfg.default_hooks.logger.interval,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )
    cfg.log_processor.by_epochs = False
    cfg.log_processor.window_size = calculate_iterations(
        epochs=cfg.log_processor.window_size,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )

    #  keep phase 1 as is because it has by_epoch = false already
    cfg.param_scheduler[1].by_epoch = False
    cfg.param_scheduler[1].convert_to_iterations = False
    cfg.param_scheduler[1].T_max = calculate_iterations(
        epochs=cfg.param_scheduler[1].T_max,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )
    cfg.param_scheduler[1].begin = calculate_iterations(
        epochs=cfg.param_scheduler[1].begin,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )

    cfg.param_scheduler[1].end = calculate_iterations(
        epochs=cfg.param_scheduler[1].end,
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )

    # Validationrelated
    cfg.train_cfg.val_interval = calculate_iterations(
        epochs=cfg.train_cfg.val_interval,
        # batch_size=cfg.val_dataloader.batch_size, I think it should be relative to the training
        batch_size=cfg.train_dataloader.batch_size,
        dataset=dataset,
    )

    # Testrelated

    cfg.dump(f"{folder_path}/{filename}")


#! iteration based configs
for filename in filenames_to_train:
    filepath = os.join(path_folder_to_train, filename)
    neck, backbone, dataset = which(filepath)
    if neck == "DiffusionDet" and backbone == "swin-b" and dataset == "coco":
        change_interations(filename, path_folder_to_train, dataset)
    elif neck == "DiffusionDet" and backbone == "swin-b" and dataset == "voc":
        change_interations(filename, path_folder_to_train, dataset)
    elif neck == "DiffusionDet" and backbone == "convnext-b" and dataset == "coco":
        change_interations(filename, path_folder_to_train, dataset)
    elif neck == "DiffusionDet" and backbone == "convnext-b" and dataset == "voc":
        change_interations(filename, path_folder_to_train, dataset)

    #! detic has batchsize [4,16] in train_dataloader find out how this works and what to assign
    elif neck == "Detic" and backbone == "swin-b" and dataset == "coco":
        change_interations(filename, path_folder_to_train, dataset)
    elif neck == "Detic" and backbone == "swin-b" and dataset == "voc":
        change_interations(filename, path_folder_to_train, dataset)
    elif neck == "Detic" and backbone == "convnext-b" and dataset == "coco":
        change_interations(filename, path_folder_to_train, dataset)
    elif neck == "Detic" and backbone == "convnext-b" and dataset == "voc":
        change_interations(filename, path_folder_to_train, dataset)
