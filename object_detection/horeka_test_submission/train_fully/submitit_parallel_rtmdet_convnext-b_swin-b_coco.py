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
]

executor = submitit.AutoExecutor(folder="slurm/train_results")

jobs = []

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
        slurm_time="10:00",
        slurm_partition="accelerated,dev_accelerated,dev_accelerated-h100,accelerated-h100",
        slurm_gres=f"gpu:{cfg.num_gpus}",
        slurm_mail_user="ruben.weber@students.uni-mannheim.de",
        slurm_mail_type="END,FAIL",
    )
    executor.submit(command)
    logger.info(f"Submitted job for {cfg}")
