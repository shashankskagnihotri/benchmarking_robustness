import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_trainer import train
from new_distributed_trainer import train_with_multiple_gpus

# from mmengine.runner import Runner
from rich.traceback import install

install(show_locals=False)


# TODO logging from Simon

slurm_log_folder_path = "slurm/work_dir"

path_configs_to_train = "./configs_to_train"

folder_entry_list_configs_to_train = os.listdir(path_configs_to_train)

path_erroneous_configs = "./configs_erroneous"
path_configs_to_test = "./configs_to_test"


path_configs_to_test_weights = "./configs_to_test_weights"


weight_work_dirs_path = "./work_dirs"  #! will done models be instead sent to results? or in the slurm directory specified?
slurm_results_path = "slurm/results"


GPU_NUM = 2  #! change when know it works as expected
slurm_partition = "dev_gpu_4"


submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

jobs = []


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


for config_file in folder_entry_list_configs_to_train:
    print(f"config_file : {config_file}")
    config_path = os.path.join(path_configs_to_train, config_file)
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

                        # TODO Marker if successful completion (how to make sure that out always gets checked first and only when checked err -> should not retrain but move to test
                        if "err" in model_logfile_ending:
                            with open(
                                os.path.join(model_log_folder_path, model_logfile), "r"
                            ) as file:
                                logfile_content = file.read()
                                if "Error" in logfile_content:
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
                                elif "DUE TO TIME LIMIT" in logfile_content:
                                    print(
                                        f"Retrain {config_file} from {model_log_folder_path}"
                                    )

                                    specific_slurm_work_dir = f"{slurm_log_folder_path}/{os.path.splitext(config_file)[0]}"
                                    specific_result_dir = f"{slurm_results_path}/{os.path.splitext(config_file)[0]}"

                                    executor = submitit.AutoExecutor(
                                        folder=specific_slurm_work_dir
                                    )

                                    executor.update_parameters(
                                        slurm_partition=f"{slurm_partition}",
                                        slurm_gres=f"gpu:{GPU_NUM}",
                                        slurm_time="00:30:00",
                                        nodes=1,
                                        cpus_per_task=4,
                                        tasks_per_node=1,
                                        slurm_mem="10G",
                                        slurm_mail_type="NONE",
                                    )

                                    job = executor.submit(
                                        train_with_multiple_gpus,  #! erronous??
                                        # train,
                                        config_path,
                                        specific_result_dir,  #! worked with train
                                        GPU_NUM,  #! not in train
                                        "auto_scale_lr",
                                    )
                        # Todo fake success -> verify
                        elif "out" in model_logfile_ending:
                            with open(
                                os.path.join(model_log_folder_path, model_logfile), "r"
                            ) as file:
                                logfile_content = file.read()
                                if "Job completed successfully" in logfile_content:
                                    print(
                                        f"moving {config_file} to {path_configs_to_test} folder"
                                    )
                                    if os.path.exists(config_path):
                                        shutil.move(
                                            config_path,
                                            os.path.join(
                                                path_configs_to_test,
                                                config_file,
                                            ),
                                        )
                                    # TODO moving weights? can just take them from the folder for testing
                                    slurm_result_folder = os.path.join(
                                        slurm_results_path,
                                        os.path.splitext(config_file)[0],
                                    )
                                    for model_weight_folder_name in slurm_result_folder:
                                        if (
                                            config_file.split(".")[0]
                                            in model_weight_folder_name
                                        ):
                                            model_weight_folder_path = os.path.join(
                                                weight_work_dirs_path,
                                                model_weight_folder_name,
                                            )
                                            highest_job_number_for_model_weights = (
                                                highest_job_number(
                                                    model_weight_folder_path
                                                )
                                            )
                                            # ? where is weight, where is mmdetection_log
    else:
        print("implement training")

# jobs.append(job)
# outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]
