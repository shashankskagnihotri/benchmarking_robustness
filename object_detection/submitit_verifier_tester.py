import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_tester import tester

# from new_distributed_tester import test_with_multiple_gpus
from new_distributed_tester_v2_fix_dist_tester import test_with_multiple_gpus


# GPU_NUM = 2
GPU_NUM = 1

# slurm_partition = "dev_gpu_4_a100" #! does not work?!?!?
slurm_partition = "dev_gpu_4"

#! updated the tester (non fix file)
#! Job 23609927 mit distributed_tester
#! Job 23610004 mit distributed_tester_v2
# config_file = "mmdetection/configs/atss/atss_r50_fpn_1x_coco.py"
# check_config_file = "slurm/work_dir/0_verification_submitit_verifier_trainer_tester/tester/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth"

#! checking if the new voc implementation works while testing
#! Job 23764103 -> was a success
config_file = "slurm/work_dir/0_verification_submitit_verifier_trainer_tester/trainer/to_check_if_epochs_repeated_atss_r50_voc0712/to_check_if_epochs_repeated_atss_r50_voc0712.py"
check_config_file = "slurm/work_dir/0_verification_submitit_verifier_trainer_tester/trainer/to_check_if_epochs_repeated_atss_r50_voc0712/epoch_1.pth"


slurm_log_folder_path = (
    "slurm/work_dir/0_verification_submitit_verifier_trainer_tester/tester"
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


#! AutoExecutor(cluster="debug") -> set break point self

#! worked with this setup
executor.submit(
    tester,
    config_file,
    check_config_file,
    specific_slurm_work_dir,
)


# executor.submit(
#     test_with_multiple_gpus,
#     config_file,
#     check_config_file,
#     specific_slurm_work_dir,
#     GPU_NUM,
# )
