import os
import shutil
from mmengine.config import Config
import submitit

from new_distributed_trainer import train_with_multiple_gpus


# from mmengine.runner import Runner
from rich.traceback import install

install(show_locals=False)

print("running config_trainer.py")

slurm_log_folder_path = "./slurm/train_work_dir"  #
slurm_log_folders = os.listdir(slurm_log_folder_path)

slurm_results_path = "slurm/train_results"
path_erroneous_configs = "./configs_erroneous/training"


# path_verified_configs = "./configs_verified" #! do this when all configs are verified
path_verified_configs = "./configs_rpn_verified"  #! currently only for rpn configs

folder_entry_list_configs_to_train = os.listdir(path_verified_configs)


path_configs_to_test = "./configs_to_test"


#! wandb vis gets currently implemented into config files!!!!!

verify_subset = False
GPU_NUM = 1
TIME = "20:00:00"
SLURM_PARTITION = "gpu_4_a100"


submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []


if verify_subset:
    folder_entry_list_configs_to_train = [
        folder_entry_list_configs_to_train[0],
        folder_entry_list_configs_to_train[1],
    ]
    path_configs_to_test = "./configs_to_test_subset"


print(f"folder_entry_list_configs_to_train : {folder_entry_list_configs_to_train}")
print(f"path_configs_to_test : {path_configs_to_test}")


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


def has_been_completed(config_file, slurm_log_folders):
    if has_been_started_before(config_file, slurm_log_folders):
        for model_log_folder_name in slurm_log_folders:
            if config_file.split(".")[0] == model_log_folder_name:
                model_log_folder_path = os.path.join(
                    slurm_log_folder_path, model_log_folder_name
                )
                model_logfiles = os.listdir(model_log_folder_path)
                highest_job_number_for_model = highest_job_number(model_log_folder_path)
                for model_logfile in model_logfiles:
                    if highest_job_number_for_model in model_logfile:
                        model_logfile_ending = model_logfile.split(".")[1]
                        if "err" in model_logfile_ending:
                            with open(
                                os.path.join(model_log_folder_path, model_logfile), "r"
                            ) as file:
                                logfile_content = file.read()
                                if "Job completed successfully" in logfile_content:
                                    print(
                                        f"moving {config_file} to {path_configs_to_test} folder"
                                    )
                                    config_path = os.path.join(
                                        path_verified_configs, config_file
                                    )

                                    if os.path.exists(config_path):
                                        shutil.move(
                                            config_path,
                                            os.path.join(
                                                path_configs_to_test,
                                                config_file,
                                            ),
                                        )
                                        folder_entry_list_configs_to_train.remove(
                                            config_file
                                        )


for config_file in folder_entry_list_configs_to_train:
    has_been_completed(config_file, slurm_log_folders)

for config_file in folder_entry_list_configs_to_train:
    print(f"config_file : {config_file}")
    config_path = os.path.join(path_verified_configs, config_file)
    print(f"config_path : {config_path}")

    if "swin-b" or "convnext-b" in config_file:  #! only train swin and convnext
        if has_been_started_before(config_file, slurm_log_folders):
            for model_log_folder_name in slurm_log_folders:
                if config_file.split(".")[0] == model_log_folder_name:
                    model_log_folder_path = os.path.join(
                        slurm_log_folder_path, model_log_folder_name
                    )
                    print(
                        f"Checking folderpath {model_log_folder_path} for {config_file}"
                    )

                    highest_job_number_for_model = highest_job_number(
                        model_log_folder_path
                    )
                    print(f"Highest job number {highest_job_number_for_model}")

                    model_logfiles = os.listdir(model_log_folder_path)

                    for model_logfile in model_logfiles:
                        if highest_job_number_for_model in model_logfile:
                            print(f"Checking logfile {model_logfile}")
                            model_logfile_ending = model_logfile.split(".")[1]

                            if "err" in model_logfile_ending:
                                print(f"checking Errorfile of {model_logfile}")
                                with open(
                                    os.path.join(model_log_folder_path, model_logfile),
                                    "r",
                                ) as file:
                                    logfile_content = file.read()
                                    if (
                                        "DUE TO TIME LIMIT" in logfile_content
                                        or "min(max_walltime * 0.8, max_walltime - 10 * 60"
                                        in logfile_content
                                        or "TypeError: can't multiply sequence by non-int of type 'float' in <mmengine.hooks.runtime_info_hook.RuntimeInfoHook"
                                        in logfile_content
                                    ):
                                        print(
                                            f"training {config_file} from {config_path}"
                                        )
                                        specific_slurm_work_dir = f"{slurm_log_folder_path}/{os.path.splitext(config_file)[0]}"
                                        specific_slurm_result_dir = f"{slurm_results_path}/{os.path.splitext(config_file)[0]}"

                                        executor = submitit.AutoExecutor(
                                            folder=specific_slurm_work_dir
                                        )

                                        executor.update_parameters(
                                            timeout_min=1,
                                            slurm_partition=f"{SLURM_PARTITION}",
                                            slurm_gres=f"gpu:{GPU_NUM}",
                                            slurm_time=f"{TIME}",
                                        )

                                        cfg = Config.fromfile(config_path)
                                        cfg.visualizer.vis_backends[
                                            0
                                        ].type = "WandbVisBackend"
                                        neck, backbone, dataset = namefinder(
                                            config_file
                                        )
                                        cfg.visualizer.vis_backends[
                                            0
                                        ].init_kwargs = dict(
                                            project=f"{neck}_{backbone}_{dataset}_train"
                                        )

                                        executor.submit(
                                            train_with_multiple_gpus,
                                            config_path,
                                            specific_slurm_result_dir,
                                            GPU_NUM,
                                        )
                                    else:
                                        # elif "Error" in logfile_content:
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
                slurm_partition=f"{SLURM_PARTITION}",
                slurm_gres=f"gpu:{GPU_NUM}",
                slurm_time=f"{TIME}",
            )

            cfg = Config.fromfile(config_path)
            cfg.visualizer.vis_backends[0].type = "WandbVisBackend"
            neck, backbone, dataset = namefinder(config_file)
            cfg.visualizer.vis_backends[0].init_kwargs = dict(
                project=f"{neck}_{backbone}_{dataset}_train"
            )

            executor.submit(
                train_with_multiple_gpus,
                config_path,
                specific_slurm_result_dir,
                GPU_NUM,
            )
