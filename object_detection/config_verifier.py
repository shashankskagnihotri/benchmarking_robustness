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

path_erroneous_configs = "./configs_erroneous/verification"
path_verified_configs = "./configs_verified"

slurm_results_path = "slurm/results"

verify_subset = False
# one_epoch = False
GPU_NUM = 2
slurm_partition = "gpu_4_a100"


submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []


if verify_subset:
    folder_entry_list_configs_to_train = [
        folder_entry_list_configs_to_train[33],  # retinanet_convnext-b_voc0712.py
        folder_entry_list_configs_to_train[38],  # retinanet_convnext-b_coco.py
    ]


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
        if config_file.split(".")[0] == model_log_folder_name:
            return True
    return False


for config_file in folder_entry_list_configs_to_train:
    print(f"config_file : {config_file}")
    config_path = os.path.join(path_configs_to_train, config_file)
    print(f"config_path : {config_path}")

    slurm_log_folders = os.listdir(slurm_log_folder_path)

    if has_been_started_before(config_file, slurm_log_folders):
        for model_log_folder_name in slurm_log_folders:
            if config_file.split(".")[0] == model_log_folder_name:
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
                                if (
                                    "DUE TO TIME LIMIT" in logfile_content
                                    or "min(max_walltime * 0.8, max_walltime - 10 * 60"
                                    in logfile_content
                                    or "can't multiply sequence by non-int of type 'float' in <mmengine.hooks.runtime"
                                    in logfile_content
                                ):
                                    #! new error handling if permitted
                                    # if (
                                    #     "DUE TO TIME LIMIT" in logfile_content
                                    #     or "min(max_walltime * 0.8, max_walltime - 10 * 60"
                                    #     in logfile_content or "can't multiply sequence by non-int of type 'float' in <mmengine.hooks.runtime" in logfile_content
                                    # ):
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
                                else:
                                    # elif (
                                    #     "Error" in logfile_content
                                    # ):  #! or "AssertionError" in logfile_content maybe just use else
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
