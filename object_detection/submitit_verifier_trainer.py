import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_trainer import trainer
from new_distributed_trainer import train_with_multiple_gpus

# GPU_NUM = 2
GPU_NUM = 1
# slurm_partition = "dev_gpu_4_a100" #! does not work?!?!?
slurm_partition = "dev_gpu_4"


#! had can't multiply sequence by non-int of type 'float' error in RuntimeInfoHook -> atss_r50_voc0712 job 23610267 -> min(max_walltime * 0.8, max_walltime - 10 * 60) error
# config_file = "configs_erroneous/verification/atss_r50_voc0712.py"

#! had can't multiply sequence by non-int of type 'float' error in RuntimeInfoHook -> atss_r101_voc0712 job 23610340 -> min(max_walltime * 0.8, max_walltime - 10 * 60) error
# config_file = "configs_erroneous/verification/atss_r101_voc0712.py"

#! had can't multiply sequence by non-int of type 'float' error in RuntimeInfoHook -> libra_rcnn_convnext-b_voc0712 job 23610354 ->
config_file = "configs_erroneous/verification/libra_rcnn_convnext-b_voc0712.py"

#! had cuda out of memory error ->  yolox_r101_coco -> same error
# config_file = "configs_erroneous/verification/yolox_r101_coco.py"


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
    slurm_time="00:30:00",
)

#! WORKS WITH WALLTIME ERROR
executor.submit(
    trainer,
    config_file,
    specific_slurm_work_dir,
)

#! WORKS WITH WALLTIME ERROR
# executor.submit(
#     train_with_multiple_gpus,
#     config_file,
#     specific_slurm_work_dir,
#     GPU_NUM,
# )
