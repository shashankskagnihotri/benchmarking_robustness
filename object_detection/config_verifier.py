import os
from mmengine.config import Config
import submitit
from tqdm import tqdm

# from mmengine.runner import Runner
from rich.traceback import install

install(show_locals=False)


# TODO Slurm Error and output
# TODO mechanism to check if training is already done / failed -> submiti checkpointing
# TODO logging from Simon

original_config_files = os.listdir("./configs_to_train")
original_config_files = original_config_files[:2]  # ? for testing purposes


#! mechanism that checks if training of that config is already done / failed


for config_file in original_config_files:
    cfg = Config.fromfile(f"./configs_to_train/{config_file}")

    if cfg.train_cfg.type == "IterBasedTrainLoop":
        cfg.train_cfg.max_iters = 100
        print(f"IterBasedTrainLoop with {cfg.train_cfg.max_iters} iterations")
    elif cfg.train_cfg.type == "EpochBasedTrainLoop":
        print("EpochBasedTrainLoop")
        cfg.train_cfg.total_epochs = 1
        print(f"EpochBasedTrainLoop with {cfg.train_cfg.total_epochs} epochs")
    else:
        raise ValueError("Unknown Train Loop Type")

    cfg.dump(f"./configs_to_train_one_epoch/{config_file}")

one_epoch_configs = os.listdir("./configs_to_train_one_epoch")

# WORK_DIR = "slurm/work_dir/%j"  #! use the config name here?
# RESULT_DIR = "slurm/results/%j"
GPU_NUM = 1  #! change when know it works as expected

# executor = submitit.AutoExecutor(folder=WORK_DIR)


jobs = []

# Submit jobs for each config
for config_file in one_epoch_configs:
    config_path = os.path.join(
        "./configs_to_train_one_epoch", config_file
    )  #! use original when want to train fully

    script_path = "./mmdetection/tools/slurm_train.sh"
    CONFIG_FILE = config_path
    WORK_DIR = f"slurm/work_dir/{os.path.splitext(config_file)[0]}"
    RESULT_DIR = f"slurm/results/{os.path.splitext(config_file)[0]}"

    executor = submitit.AutoExecutor(folder=WORK_DIR)

    #! change settings for slurm
    executor.update_parameters(
        slurm_partition="gpu_4",
        slurm_gres=f"gpu:{GPU_NUM}",
        slurm_time="02:00:00",
        nodes=1,
        cpus_per_task=16,
        tasks_per_node=1,
        slurm_mem="20G",
        slurm_mail_type="ALL",
        slurm_mail_user="ruben.weber@students.uni-mannheim.de",
        # slurm_error=os.path.join(RESULT_DIR, "errors"),
        # slurm_output=os.path.join(RESULT_DIR, "output"),
    )
    submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

    cmd = f"bash {script_path} {config_path} {GPU_NUM} --work_dir {WORK_DIR}"

    job = executor.submit(submitit.helpers.CommandFunction([cmd]))
    jobs.append(job)

# Wait for all jobs to finish needed?
# results = executor.wait()

#! gather results or perform other actions
outputs = [job.result() for job in tqdm(jobs, desc="Collecting Results")]


#! try with config file name as differentiator
# import os
# from mmengine.config import Config
# import submitit
# from tqdm import tqdm
# from rich.traceback import install

# install(show_locals=False)

# original_config_files = os.listdir("./configs_to_train")
# original_config_files = original_config_files[:2]

# for config_file in original_config_files:
#     cfg = Config.fromfile(f"./configs_to_train/{config_file}")
#     cfg.total_epochs = 1
#     cfg.dump(f"./configs_to_train_one_epoch/{config_file}")

# one_epoch_configs = os.listdir("./configs_to_train_one_epoch")

# GPU_NUM = 2

# executor = submitit.AutoExecutor(folder="slurm/work_dir")  # No need to specify %j here

# executor.update_parameters(
#     slurm_partition="gpu_4",
#     slurm_gres=f"gpu:{GPU_NUM}",
#     slurm_time="02:00:00",
#     nodes=1,
#     cpus_per_task=16,
#     tasks_per_node=1,
#     slurm_mem="20G",
#     slurm_mail_type="ALL",
#     slurm_mail_user="ruben.weber@students.uni-mannheim.de",
# )

# submitit.helpers.CommandFunction(["module", "load", "devel/cuda/11.8"])

# jobs = []

# # Submit jobs for each config
# for config_file in one_epoch_configs:
#     config_path = os.path.join("./configs_to_train_one_epoch", config_file)
#     CONFIG_FILE = config_path

#     # Use the config filename as the directory name
#     WORK_DIR = f"slurm/work_dir/{config_file.split('.')[0]}"

#     cmd = f"bash ./tools/dist_train.sh {CONFIG_FILE} {GPU_NUM} --work_dir {WORK_DIR}"

#     job = executor.submit(submitit.helpers.CommandFunction([cmd]))
#     jobs.append(job)

# # Wait for all jobs to finish
# results = executor.wait()

# # Gather results or perform other actions
# outputs = [job.result() for job in tqdm(jobs, desc="Collecting Results")]


# for configs_train in os.listdir("./configs_to_train"):
#     print(configs_train)
#     cfg = Config.fromfile(f"./configs_to_train/{configs_train}")
#     cfg.work_dir = "./work_dirs/"
#     cfg.total_epochs = 1
#     runner = Runner.from_cfg(cfg)
#     runner.train()


#! something to check if done correctly -> move into other folder?
#! and something if fails -> stays in folder but how that it does not always runs the same failing ones

#! slurm per python bei Jonas schauen in submit attacks
#! braucht ne function also training als function machen und slurm übergeben
#! wenn slurm error macht dann irgendwo saven
