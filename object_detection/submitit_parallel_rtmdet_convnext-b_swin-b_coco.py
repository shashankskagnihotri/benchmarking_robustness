from dataclasses import dataclass

import submitit

# rich logger
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)

logger = logging.getLogger()


@dataclass
class cfg:
    model_name: str
    num_gpus: int


cfgs = [
    cfg("rtmdet_convnext-b_coco", 1),
    cfg("rtmdet_swin-b_coco", 3),
    cfg("dino_convnext-b_coco", 2),
    # cfg("dino_swin-b_coco", 4),
]


executor = submitit.AutoExecutor(folder="slurm/train_results")

jobs = []

# accelerated-h100
# accelerated

#
# cfg("dino_swin-b_coco", 4), accelerated, 2669626_submission -> memory error
# cfg("dino_swin-b_coco", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device

# cfg("rtmdet_convnext-b_coco", 1), accelerated-h100, 2669839_submitted ->
# cfg("rtmdet_swin-b_coco", 3), accelerated-h100, 2669840_submission ->
# cfg("dino_convnext-b_coco", 2), accelerated-h100, 2669841_submission ->

for cfg in cfgs:
    command = submitit.helpers.CommandFunction(
        [
            "mmdetection/tools/dist_train.sh",
            f"./horeka_test_submission/train_fully/{cfg.model_name}.py",
            str(cfg.num_gpus),
            "--work-dir",
            f"./slurm/train_work_dir/{cfg.model_name}",
            "--resume",
            "--auto-scale-lr",
        ],
        verbose=True,
    )

    executor.update_parameters(
        tasks_per_node=1,
        nodes=1,
        slurm_time="2-00:00:00",
        slurm_partition="accelerated-h100",
        slurm_gres=f"gpu:{cfg.num_gpus}",
        slurm_mail_user="ruben.weber@students.uni-mannheim.de",
        slurm_mail_type="END,FAIL",
    )
    executor.submit(command)
    logger.info(f"Submitted job for {cfg}")
