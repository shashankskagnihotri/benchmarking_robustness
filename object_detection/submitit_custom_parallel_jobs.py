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
    # cfg("rtmdet_convnext-b_coco", 1),
    # cfg("rtmdet_swin-b_coco", 3),
    # cfg("dino_convnext-b_coco", 2),
    # cfg("dino_swin-b_coco", 4),
    cfg("dino_convnext-s_coco", 2),
    cfg("dino_swin-s_coco", 2),
]


executor = submitit.AutoExecutor(folder="slurm/train_results")

jobs = []

# accelerated-h100
# accelerated

# TODO: gpu_assign_thr=N
# TODO: Switch to pascal dataset


#! swin-s and convnext-s runs
#! cfg("dino_convnext-s_coco", 1), accelerated 2672792 -> RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to listen on [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to 0.0.0.0:29500 (errno: 98 - Address already in use).
#! cfg("dino_swin-s_coco", 1), accelerated 2672793 -> torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 102.00 MiB. GPU 0 has a total capacty of 39.38 GiB of which 29.81 MiB is free. Including non-PyTorch memory, this process has 39.32 GiB memory in use. Of the allocated memory 37.54 GiB is allocated by PyTorch, and 1.12 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# cfg("dino_convnext-s_coco", 1) accelerated 2674043 ->
# cfg("dino_swin-s_coco", 1) accelerated 2674044 ->


# cfg("dino_swin-b_coco", 4), accelerated, 2669626_submission -> memory error
# cfg("dino_swin-b_coco", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device

#! cfg("rtmdet_convnext-b_coco", 1), accelerated-h100, 2669839_submitted -> running till val loop than crashed because of CUDA error: no kernel image is available for execution on the device
#! cfg("rtmdet_swin-b_coco", 3), accelerated-h100, 2669840_submission -> running till val loop than crashed because of CUDA error: no kernel image is available for execution on the device
# cfg("dino_convnext-b_coco", 2), accelerated-h100, 2669841_submission -> memory error and child process exited

# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in train.py v1, 2671230_submission -> RuntimeError: `scale_lr` should be called before building ParamScheduler because ParamScheduler will store initial lr from optimizer wrappers
# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in train.py v2, 2671256_submission -> RuntimeError: `scale_lr` should be called before building ParamScheduler because ParamScheduler will store initial lr from optimizer wrappers
# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in distributed.py, 2671264_submission ->     raise ValueError('There is no tensor in the inputs, 'ValueError: There is no tensor in the inputs, cannot get dtype and device.
# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in distributed.py with type to tensor changing, 2671284_submission ->     raise ValueError('There is no tensor in the inputs, 'ValueError: There is no tensor in the inputs, cannot get dtype and device.


# cfg("dino_swin-b_voc0712", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device
# cfg("dino_swin-b_voc0712", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device


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
        slurm_partition="accelerated",
        slurm_gres=f"gpu:{cfg.num_gpus}",
        slurm_mail_user="ruben.weber@students.uni-mannheim.de",
        slurm_mail_type="END,FAIL",
    )
    executor.submit(command)
    logger.info(f"Submitted job for {cfg}")
