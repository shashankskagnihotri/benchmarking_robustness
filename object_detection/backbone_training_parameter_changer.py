import os
from mmengine.config import Config
from config_maker import config_keybased_value_changer, adjust_param_scheduler, which


#! check if it works properly


#! optimizer (lr, weight decay etc), paramscheduler, epochs, batchsize, autoscale, scheduling, expMomentumEMA?

#! same for voc use the functions (config_keybased_value_changer, adjust_param_scheduler) out of configmaker to change in right way

# Load reference configs
swin_b_cfg = Config.fromfile("rtmdet_swin_b_coco.py")
convnext_b_cfg = Config.fromfile("rtmdet_convnext_b_coco.py")

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)


def change_training_implementation(filename, folder_path, backbone_cfg):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    cfg.optim_wrapper = backbone_cfg.optim_wrapper
    cfg.param_scheduler = backbone_cfg.param_scheduler
    cfg.max_epochs = backbone_cfg.max_epochs
    if (
        cfg.train_cfg.type == "EpochBasedTrainLoop"
    ):  #!!!! how to handle iteration based?
        cfg.train_cfg.max_epochs = backbone_cfg.train_cfg.max_epochs
    #! look into individual cfg.train_cfg and depending on implementation import full cfg.train_cfg from backbone_cfg or keep some parts (iterationbased, _scope_, dynamic intervals -> change the iterations based on epochs etc)

    # cfg.custom_hooks = backbone_cfg.custom_hooks[0] #! only use exp_ema and wait for call of tutor regarding the keeping of yolox custom hooks

    if hasattr(cfg, "custom_hooks") and cfg.custom_hooks:
        print(f"Custom hooks in {filename} found:")
        for hook in cfg.custom_hooks:
            print(hook)

    cfg.default_hooks.param_scheduler = backbone_cfg.default_hooks.param_scheduler
    cfg.deault_hooks.sampler_seed = backbone_cfg.default_hooks.sampler_seed

    cfg.train_dataloader.batch_sample = backbone_cfg.train_dataloader.batch_sampler
    cfg.train_dataloader.batch_size = backbone_cfg.train_dataloader.batch_size
    cfg.train_dataloader.num_workers = backbone_cfg.train_dataloader.num_workers
    cfg.train_dataloader.persistent_workers = (
        backbone_cfg.train_dataloader.persistent_workers
    )

    if hasattr(cfg, "auto_scale_lr"):
        cfg.auto_scale_lr.enable = True
    else:
        cfg.auto_scale_lr = dict(enable=True)

    cfg.dump(
        f"{folder_path}/{filename}"
    )  #! maybe all to training such that retest or only test on subset


for filename in filenames_to_train:
    if "swin_b" in filename:
        change_training_implementation(filename, path_folder_to_train, swin_b_cfg)
    elif "convnext_b" in filename:
        change_training_implementation(filename, path_folder_to_train, convnext_b_cfg)

for filename in filenames_verified:
    if "swin_b" in filename:
        change_training_implementation(filename, path_folder_verified, swin_b_cfg)
    elif "convnext_b" in filename:
        change_training_implementation(filename, path_folder_verified, convnext_b_cfg)

for filename in filenames_erroneous:
    if "swin_b" in filename:
        change_training_implementation(filename, path_folder_erroneous, swin_b_cfg)
    elif "convnext_b" in filename:
        change_training_implementation(filename, path_folder_erroneous, convnext_b_cfg)

for filename in filenames_to_test:
    if "swin_b" in filename:
        change_training_implementation(filename, path_folder_to_test, swin_b_cfg)
    elif "convnext_b" in filename:
        change_training_implementation(filename, path_folder_to_test, convnext_b_cfg)


#! find all parts, where iterations are needed and change them to epochs
#! function which calculates needed iterations based on batchsize, dataset size and epochs


def calculate_iterations(epochs, batch_size, dataset):
    #! cfg.train_dataloader.batch_size
    #! epochs the depending cfg. ....
    if dataset == "coco":
        dataset_size = 0  #! get the right absolute dataset size
    elif dataset == "voc":
        dataset_size = 0  #! get the right absolute dataset size

    steps_per_epoch = dataset_size / batch_size
    total_iterations = epochs * steps_per_epoch

    return int(total_iterations)


#! iteration based configs
for filename in filenames_to_train:
    filepath = os.join(path_folder_to_train, filename)
    neck, backbone, dataset = which(filepath)
# if neck == "DiffusionDet" and backbone =="swin-b" and dataset == "coco":
# cfg = Config.fromfile(filepath)
# .... iterationstuff
# elif neck == "DiffusionDet" and backbone =="swin-b" and dataset == "voc":
# .... iterationstuff
# elif neck == "DiffusionDet" and backbone =="convnext-b" and dataset == "coco":
# .... iterationstuff
# elif neck == "DiffusionDet" and backbone =="convnext-b" and dataset == "voc":
# ....
# elif neck == "Detic" and backbone =="swin-b" and dataset == "coco":
# ....
# elif neck == "Detic" and backbone =="swin-b" and dataset == "voc":
# ....
# elif neck == "Detic" and backbone =="convnext-b" and dataset == "coco":
# ....
# elif neck == "Detic" and backbone =="convnext-b" and dataset == "voc":
# ....
