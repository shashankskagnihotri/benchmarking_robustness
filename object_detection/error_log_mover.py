import os
import shutil
from config_verifier import highest_job_number


path_slurm_log_files_folder = "slurm/work_dir"
folders_slurm_log_files = os.listdir(path_slurm_log_files_folder)


path_erroneous_config_files = "./configs_erroneous/verification"
folder_erroneous_config_files = os.listdir(path_erroneous_config_files)
log_error_path = "slurm/work_dir/log_error"

for config_file in folder_erroneous_config_files:
    config_path = os.path.join(path_erroneous_config_files, config_file)

    for model_log_folder_name in folders_slurm_log_files:
        if config_file.split(".")[0] in model_log_folder_name:
            model_log_folder_path = os.path.join(
                path_slurm_log_files_folder, model_log_folder_name
            )
            model_logfiles = os.listdir(model_log_folder_path)

            for model_logfile in model_logfiles:
                if highest_job_number(model_log_folder_path) in model_logfile:
                    if "err" in model_logfile.split(".")[1]:
                        with open(
                            os.path.join(model_log_folder_path, model_logfile), "r"
                        ) as file:
                            logfile_content = file.read()

                            if (
                                "DUE TO TIME LIMIT" in logfile_content
                                or "min(max_walltime * 0.8, max_walltime - 10 * 60"
                                in logfile_content
                            ):
                                pass

                            else:
                                # dump the model_logfile variable into the log_error folder (path = log_error_path) under the name in the model_log_folder_name variable + .err
                                print(
                                    f"copying {model_logfile} to {os.path.basename(model_log_folder_name)} + .err folder"
                                )
                                shutil.copy(
                                    os.path.join(model_log_folder_path, model_logfile),
                                    os.path.join(
                                        log_error_path,
                                        os.path.basename(model_log_folder_name)
                                        + ".err",
                                    ),
                                )
