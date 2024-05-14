import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_trainer import trainer
from new_distributed_trainer import train_with_multiple_gpus

GPU_NUM = 2
# slurm_partition = "dev_gpu_4_a100" #! does not work?!?!?
slurm_partition = "dev_gpu_4"


#! has run properly before changes and dab_detr_r101_voc0712.py did not run properly
config_file = "configs_verified/dab_detr_r101_coco.py"
# config_file = "configs_verified/dab_detr_r101_voc0712.py"

slurm_log_folder_path = (
    "slurm/work_dir/0_verification_submitit_verifier_trainer_tester/trainer"
)
specific_slurm_work_dir = f"{slurm_log_folder_path}/{os.path.splitext(config_file)[0]}"


executor = submitit.AutoExecutor(folder=specific_slurm_work_dir)

#! works with this setup
executor.update_parameters(
    timeout_min=1,
    slurm_partition=f"{slurm_partition}",
    slurm_gres=f"gpu:{GPU_NUM}",
    slurm_time="00:10:00",
)

# executor.submit(
#     trainer,
#     config_file,
#     specific_slurm_work_dir,
# )

#! 23598515
executor.submit(
    train_with_multiple_gpus,
    config_file,
    specific_slurm_work_dir,
    GPU_NUM,
)
