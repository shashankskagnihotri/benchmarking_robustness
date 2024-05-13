import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_trainer import trainer
from new_distributed_trainer import train_with_multiple_gpus

# from mmengine.runner import Runner
from rich.traceback import install

install(show_locals=False)


# TODO logging from Simon


slurm_log_folder_path = "slurm/work_dir"


path_configs_to_train = "./configs_to_train"

folder_entry_list_configs_to_train = os.listdir(path_configs_to_train)

folder_entry_list_one_epoch_configs = os.listdir("./configs_to_verify_one_epoch")

path_configs_to_verify_one_epoch = "./configs_to_verify_one_epoch"
path_erroneous_configs = "./configs_erroneous"
path_verified_configs = "./configs_verified"

weight_work_dirs_path = "./work_dirs"  #! will done models be instead sent to results or in the slurm workdir?
slurm_results_path = "slurm/results"

verify_subset = False
# one_epoch = False
GPU_NUM = 1
slurm_partition = "gpu_4"


submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []


if verify_subset:
    folder_entry_list_configs_to_train = [
        folder_entry_list_configs_to_train[33],  # retinanet_convnext-b_voc0712.py
        folder_entry_list_configs_to_train[38],  # retinanet_convnext-b_coco.py
    ]


# if one_epoch:

# for config_file in folder_entry_list_configs_to_train:
#     cfg = Config.fromfile(os.path.join(path_configs_to_train, config_file))

#     if cfg.train_cfg.type == "IterBasedTrainLoop":
#         cfg.train_cfg.max_iters = 1000
#         print(f"IterBasedTrainLoop with {cfg.train_cfg.max_iters} iterations")
#     elif cfg.train_cfg.type == "EpochBasedTrainLoop":
#         print("EpochBasedTrainLoop")
#         cfg.train_cfg.max_epochs = 1
#         print(f"EpochBasedTrainLoop with {cfg.train_cfg.max_epochs} epochs")
#     else:
#         raise ValueError("Unknown Train Loop Type")

#     cfg.dump(os.path.join(path_configs_to_verify_one_epoch, config_file))
#     path_configs_to_verify_one_epoch = path_configs_to_train

# else:
#   path_configs_to_verify_one_epoch = path_configs_to_train


#! 23562330 atss_convnext-b_coco.py -> submitit timeout

# ? are run later remove one of those and replace by 23562330
# ? 23562468
# ? 23562467
path_configs_to_verify_one_epoch = path_configs_to_train
folder_entry_list_one_epoch_configs = folder_entry_list_configs_to_train

print(path_configs_to_verify_one_epoch)
print(folder_entry_list_one_epoch_configs)


def highest_job_number(model_log_folder_path):
    model_logfiles = os.listdir(model_log_folder_path)
    job_numbers = set()

    for logfile in model_logfiles:
        job_number = logfile.split("_")[0]
        if job_number.isdigit():
            job_numbers.add(job_number)
    highest_job_number = max(job_numbers)
    return highest_job_number


def has_been_started_before(config_file, slurm_log_folders):
    for model_log_folder_name in slurm_log_folders:
        if config_file.split(".")[0] in model_log_folder_name:
            return True
    return False


for config_file in folder_entry_list_one_epoch_configs:
    print(f"config_file : {config_file}")
    config_path = os.path.join(path_configs_to_verify_one_epoch, config_file)
    print(f"config_path : {config_path}")

    slurm_log_folders = os.listdir(slurm_log_folder_path)

    if has_been_started_before(config_file, slurm_log_folders):
        for model_log_folder_name in slurm_log_folders:
            if config_file.split(".")[0] in model_log_folder_name:
                model_log_folder_path = os.path.join(
                    slurm_log_folder_path, model_log_folder_name
                )
                print(f"Checking folderpath {model_log_folder_path} for {config_file}")

                highest_job_number_for_model = highest_job_number(model_log_folder_path)
                print(f"Highest job number {highest_job_number_for_model}")

                model_logfiles = os.listdir(model_log_folder_path)

                for model_logfile in model_logfiles:
                    if highest_job_number_for_model in model_logfile:
                        print(f"Checking logfile {model_logfile}")
                        model_logfile_ending = model_logfile.split(".")[1]

                        if "err" in model_logfile_ending:
                            print(f"checking Errorfile of {model_logfile}")
                            with open(
                                os.path.join(model_log_folder_path, model_logfile), "r"
                            ) as file:
                                logfile_content = file.read()
                                if "DUE TO TIME LIMIT" in logfile_content:
                                    print(
                                        f"moving {config_file} to {path_verified_configs} folder"
                                    )

                                    if os.path.exists(config_path):
                                        shutil.move(
                                            config_path,
                                            os.path.join(
                                                path_verified_configs,
                                                config_file,
                                            ),
                                        )
                                elif "Error" in logfile_content:
                                    print(
                                        f"moving {config_file} to {path_erroneous_configs} folder"
                                    )
                                    if os.path.exists(config_path):
                                        shutil.move(
                                            config_path,
                                            os.path.join(
                                                path_erroneous_configs, config_file
                                            ),
                                        )

    else:
        print(f"training {config_file} from {config_path}")
        specific_slurm_work_dir = (
            f"{slurm_log_folder_path}/{os.path.splitext(config_file)[0]}"
        )
        specific_slurm_result_dir = (
            f"{slurm_results_path}/{os.path.splitext(config_file)[0]}"
        )

        executor = submitit.AutoExecutor(folder=specific_slurm_work_dir)

        executor.update_parameters(
            timeout_min=1,
            slurm_partition=f"{slurm_partition}",
            slurm_gres=f"gpu:{GPU_NUM}",
            slurm_time="00:10:00",
        )
        executor.submit(
            trainer,
            config_path,
            specific_slurm_result_dir,
        )


# srun: error: CPU binding outside of job step allocation, allocated CPUs are: 0x00C0001C0000C0001C00.
# srun: error: Task launch for StepId=23562467.0 failed on node uc2n520: Unable to satisfy cpu bind request
# srun: error: Application launch failed: Unable to satisfy cpu bind request
# srun: Job step aborted
