import os
import shutil
from mmengine.config import Config
import submitit
from tqdm import tqdm

from new_trainer import trainer
from new_distributed_trainer import train_with_multiple_gpus

# GPU_NUM = 2
GPU_NUM = 1
# slurm_partition = "dev_gpu_4_a100"  #! does not work?!?!?
slurm_partition = "gpu_4"


#! had can't multiply sequence by non-int of type 'float' error in RuntimeInfoHook -> atss_r50_voc0712 job 23610267 -> min(max_walltime * 0.8, max_walltime - 10 * 60) error
# config_file = "configs_erroneous/verification/atss_r50_voc0712.py"

#! had can't multiply sequence by non-int of type 'float' error in RuntimeInfoHook -> atss_r101_voc0712 job 23610340 -> min(max_walltime * 0.8, max_walltime - 10 * 60) error
# config_file = "configs_erroneous/verification/atss_r101_voc0712.py"

#! had can't multiply sequence by non-int of type 'float' error in RuntimeInfoHook -> libra_rcnn_convnext-b_voc0712 job 23615948 ->
# config_file = "configs_erroneous/verification/libra_rcnn_convnext-b_voc0712.py"

#! had cuda out of memory error ->  yolox_r101_coco -> same error
# config_file = "configs_erroneous/verification/yolox_r101_coco.py"


# config_file = "configs_verified/dab_detr_r101_voc0712.py"

#! to check if repeate datasets actually repeats a epoch -> max_epochs set to 1 should run for 3 epochs
#! might be under 23720087_submission -> unfortunaly cancled with scancel -u $USER
#! new run to obtain weights 23737571 -> did run 3 times
# config_file = "./to_check_if_epochs_repeated_atss_r50_voc0712.py"


# EfficientDet r50, r101 have memory error. Wasn´t fixed by one a100 now testing two a100s
# configs_erroneous/verification/EfficientDet_r50_coco.py -> 23909208
# configs_erroneous/verification/EfficientDet_r101_coco.py -> 23909211
# configs_erroneous/verification/EfficientDet_swin-b_coco.py -> 23909245

config_file = "configs_verified/centernet_r50_voc0712_vocmetric.py"

slurm_log_folder_path = (
    "slurm/work_dir/0_verification_submitit_verifier_trainer_tester/trainer"
)
specific_slurm_work_dir = f"{slurm_log_folder_path}/{os.path.splitext(config_file)[0]}"

# current job 23871411

# voc vs cocometric only changed voc/cocometric eveything else is the same
# vocmetric job: 23875097, 23877306, 23880820, 23881551, 23883119, 23886921, 23909263
# cocometric job: 23875095, 23877307, 23880853, 23883120, 23886920,
# vocmetric 3x job: 23892625,
# cocometric 3x job: 23892618,

executor = submitit.AutoExecutor(folder=specific_slurm_work_dir)

#! works with this setup
executor.update_parameters(
    # timeout_min=5, might cause conflicts with slurm_time
    slurm_partition=f"{slurm_partition}",
    slurm_gres=f"gpu:{GPU_NUM}",
    slurm_time="15:00:00",
)

#! WORKS WITH WALLTIME ERROR
# executor.submit(
#    trainer,
#    config_file,
#    specific_slurm_work_dir,
# )

#! WORKS WITH WALLTIME ERROR
executor.submit(
    train_with_multiple_gpus,
    config_file,
    specific_slurm_work_dir,
    GPU_NUM,
)
