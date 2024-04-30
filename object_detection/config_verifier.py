import os
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_trainer import train
from new_distributed_trainer import train_with_multiple_gpus

# from mmengine.runner import Runner
from rich.traceback import install

install(show_locals=False)


# TODO mechanism to check if training is already done (skip) / started (resume) / failed (skip) -> submiti checkpointing


log_folder_path = "slurm/work_dir"

for log_folder in os.listdir(log_folder_path):
    folder_path = os.path.join(log_folder_path, log_folder)
    if os.path.isdir(folder_path):
        for logfile in os.listdir(folder_path):
            if "err" in logfile:
                with open(os.path.join(folder_path, logfile), "r") as file:
                    logfile_content = file.read()
                    if "Error" in logfile_content:
                        print(f"Error in {folder_path}")
                        #! move to different location such that not retrained
            if "out" in logfile:
                with open(os.path.join(folder_path, logfile), "r") as file:
                    logfile_content = file.read()
                    if "successful completion" in logfile_content:
                        print(f"Successful completion of {folder_path}")
                        #! move to different location such that not retrained


# TODO logging from Simon

original_config_files = os.listdir("./configs_to_train")  #! change to_train
original_config_files = [
    original_config_files[149],  # retinanet_swin-b_coco.py
    original_config_files[145],  # ddod_swin-b_coco.py
]  # ? for testing purposes

for config_file in original_config_files:
    cfg = Config.fromfile(f"./configs_to_train/{config_file}")  #! to_train

    if cfg.train_cfg.type == "IterBasedTrainLoop":
        cfg.train_cfg.max_iters = 10
        print(f"IterBasedTrainLoop with {cfg.train_cfg.max_iters} iterations")
    elif cfg.train_cfg.type == "EpochBasedTrainLoop":
        print("EpochBasedTrainLoop")
        cfg.train_cfg.max_epochs = 1
        print(f"EpochBasedTrainLoop with {cfg.train_cfg.max_epochs} epochs")
    else:
        raise ValueError("Unknown Train Loop Type")

    cfg.dump(f"./configs_to_verify_one_epoch/{config_file}")

one_epoch_configs = os.listdir("./configs_to_verify_one_epoch")

GPU_NUM = 2  #! change when know it works as expected

# Submit jobs for each config

submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []
for config_file in one_epoch_configs:
    config_path = os.path.join(
        "./configs_to_verify_one_epoch", config_file
    )  #! use original when want to train fully

    work_dir = f"slurm/work_dir/{os.path.splitext(config_file)[0]}"
    executor = submitit.AutoExecutor(folder=work_dir)

    executor.update_parameters(
        slurm_partition="dev_gpu_4",
        slurm_gres=f"gpu:{GPU_NUM}",
        slurm_time="00:30:00",
        nodes=1,
        cpus_per_task=4,
        tasks_per_node=1,
        slurm_mem="10G",
        slurm_mail_type="NONE",
    )
    result_dir = f"slurm/results/{os.path.splitext(config_file)[0]}"

    job = executor.submit(
        train_with_multiple_gpus,
        # train,
        config_path,
        GPU_NUM,  #! not in train
        # None,  # cfg_options #! worked with train
        # result_dir,  #! worked with train
    )
    jobs.append(job)


outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]
