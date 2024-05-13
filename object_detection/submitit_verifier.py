import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm


from new_trainer import trainer


GPU_NUM = 1
# slurm_partition = "dev_gpu_4_a100" #! does not work?!?!?
slurm_partition = "dev_gpu_4"


config_file = "configs_to_train/atss_convnext-b_coco.py"

slurm_log_folder_path = "slurm/work_dir"
specific_slurm_work_dir = f"{slurm_log_folder_path}/{os.path.splitext(config_file)[0]}"


executor = submitit.AutoExecutor(folder=specific_slurm_work_dir)

#! works with this setup
executor.update_parameters(
    timeout_min=1,
    slurm_partition=f"{slurm_partition}",
    slurm_gres=f"gpu:{GPU_NUM}",
    slurm_time="00:30:00",
)

# ? tested currently
#! 23562330 -> model v2


# executor.update_parameters(
#     timeout_min=1,
#     slurm_partition=f"{slurm_partition}",
#     slurm_gres=f"gpu:{GPU_NUM}",
#     slurm_time="00:30:00",
#     nodes=1,
#     cpus_per_task=4,
#     tasks_per_node=1,
#     slurm_mem="10G",
#     slurm_mail_type="NONE",
# )
#! AutoExecutor(cluster="debug") -> set break point self


# executor.submit(
#     # train,
#     trainer,
#     config_file,
#     specific_slurm_work_dir,
# )

#! works with this setup
executor.submit(
    trainer,
    config_file,
    specific_slurm_work_dir,
)


#! works
def add(a, b):
    return a + b


# job = executor.submit(add, 5, 7)  # will compute add(5, 7)
