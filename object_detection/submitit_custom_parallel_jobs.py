from dataclasses import dataclass
import random
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
    # cfg("cascade_rcnn_convnext-s_coco", 2),
    # cfg("cascade_rcnn_swin-s_coco", 2),
    # cfg("codino_convnext-s_coco", 2),
    # cfg("codino_swin-s_coco", 2),
    # cfg("ddq_convnext-s_coco", 2),
    # cfg("ddq_swin-s_coco", 2),
    # cfg("deformable_detr_convnext-s_coco", 4),
    # cfg("deformable_detr_swin-s_coco", 4),
    # cfg("dino_convnext-s_coco", 2),
    # cfg("dino_swin-s_coco", 2),
    # cfg("glip_convnext-s_coco", 2),
    # cfg("glip_swin-s_coco", 2),
    cfg("paa_convnext-s_coco", 2),  # ?
    cfg("paa_swin-s_coco", 2),  # ?
    # cfg("rtmdet_convnext-s_coco", 2),
    # cfg("rtmdet_swin-s_coco", 2),
    # cfg("sparse_rcnn_convnext-s_coco", 2),
    # cfg("sparse_rcnn_swin-s_coco", 2),
    # cfg("tood_convnext-s_coco", 2),  # ?
    # cfg("tood_swin-s_coco", 2),  # ?
]


executor = submitit.AutoExecutor(folder="slurm/train_results")

jobs = []

# accelerated-h100
# accelerated

# TODO: gpu_assign_thr=N
# TODO: Switch to pascal dataset


#! swin-s and convnext-s runs
# cfg("dino_convnext-s_coco", 1), accelerated 2672792 -> RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to listen on [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to 0.0.0.0:29500 (errno: 98 - Address already in use).
# cfg("dino_swin-s_coco", 1), accelerated 2672793 -> torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 102.00 MiB. GPU 0 has a total capacty of 39.38 GiB of which 29.81 MiB is free. Including non-PyTorch memory, this process has 39.32 GiB memory in use. Of the allocated memory 37.54 GiB is allocated by PyTorch, and 1.12 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# cfg("dino_convnext-s_coco", 2) was with convnext-b training hp accelerated 2674043 -> memory error
# cfg("dino_swin-s_coco", 2) was with swin-b training hp accelerated 2674044 -> memory error

# cfg("dino_convnext-s_coco", 4) was with convnext-b training hp accelerated 2674497 -> memory error
# cfg("dino_swin-s_coco", 4) was with swin-b training hp accelerated 2674498 -> memory error

# cfg("dino_convnext-s_coco", 1) was with convnext-s training hp accelerated 2675047 -> did run
# cfg("dino_swin-s_coco", 1) was with swin-s training hp accelerated 2675048 -> adress already in use
# cfg("dino_swin-s_coco", 1) was with swin-s training hp accelerated 2675055 -> adress already in use
# cfg("dino_swin-s_coco", 2) was with swin-s training hp accelerated 2675057 -> adress already in use
# cfg("dino_swin-s_coco", 2) was with swin-s training hp accelerated (first killed the terminal and used a new one) 2675061 -> adress already in use


# cfg("dino_swin-s_coco", 1) was with covnext-s training hp accelerated after trying port setting 2675106 -> ran
# cfg("dino_convnext-s_coco", 1) was with swin-s training hp accelerated after trying port setting 2675105 -> ran
# cfg("rtmdet_convnext-s_coco", 1) was with swin-s training hp accelerated after trying port setting 2675140 -> ValueError: num_channels must be divisible by num_groups
# cfg("rtmdet_swin-s_coco", 1) was with covnext-s training hp accelerated after trying port setting 2675141 -> ValueError: num_channels must be divisible by num_groups


# cfg("rtmdet_convnext-s_coco", 2) was with convnext-s training hp accelerated after trying port setting 2678905 -> fogot wandb
# cfg("rtmdet_swin-s_coco", 2) was with swin-s training hp accelerated after trying port setting 2678906 -> raise ValueError("some parameters appear in more than one parameter group")
# cfg("rtmdet_swin-s_coco", 2) was with swin-s training hp accelerated after trying port setting; changed paramwise... 2679458 -> fogot wandb


# * cfg("dino_convnext-s_coco", 2) 2678903 -> finished due tue early stopping with patience 5; Epoch(val) [10][2500/2500]    coco/bbox_mAP: 0.4870  coco/bbox_mAP_50: 0.6670  coco/bbox_mAP_75: 0.5300  coco/bbox_mAP_s: 0.3260  coco/bbox_mAP_m: 0.5190  coco/bbox_mAP_l: 0.6310
# * cfg("dino_swin-s_coco", 2) 2678904 -> finished due tue early stopping with patience 5; Epoch(val) [10][2500/2500]    coco/bbox_mAP: 0.4760  coco/bbox_mAP_50: 0.6570  coco/bbox_mAP_75: 0.5180  coco/bbox_mAP_s: 0.3080  coco/bbox_mAP_m: 0.5080  coco/bbox_mAP_l: 0.6230
# * cfg("cascade_rcnn_convnext-s_coco", 2), 2679970 -> finished due tue early stopping with patience 5; Epoch(val) [6][2500/2500]    coco/bbox_mAP: 0.4400  coco/bbox_mAP_50: 0.6400  coco/bbox_mAP_75: 0.4830  coco/bbox_mAP_s: 0.2680  coco/bbox_mAP_m: 0.4780  coco/bbox_mAP_l: 0.5840
# * cfg("cascade_rcnn_swin-s_coco", 2), 2679971 -> finished due tue early stopping with patience 5; Epoch(val) [6][2500/2500]    coco/bbox_mAP: 0.4490  coco/bbox_mAP_50: 0.6500  coco/bbox_mAP_75: 0.4870  coco/bbox_mAP_s: 0.2710  coco/bbox_mAP_m: 0.4830  coco/bbox_mAP_l: 0.6060
#  cfg("deformable_detr_convnext-s_coco", 2), 2679972 -> out of memory
#  cfg("deformable_detr_swin-s_coco", 2), 2679973 -> out of memory
# cfg("glip_convnext-s_coco", 2), 2679974 -> resubmit
# cfg("glip_swin-s_coco", 2), 2679975 -> resubmit
# cfg("rtmdet_convnext-s_coco", 2), 2679976 -> resubmit
# cfg("rtmdet_swin-s_coco", 2), 2679977 ->


#  cfg("deformable_detr_convnext-s_coco", 4), 2680326 -> out of memory
#  cfg("deformable_detr_swin-s_coco", 4), 2680327 -> out of memory

# * cfg("deformable_detr_convnext-s_coco", 4), with mixed precision 2680359  -> finished due tue early stopping with patience 5; Epoch(val) [8][1250/1250]    coco/bbox_mAP: 0.3990  coco/bbox_mAP_50: 0.5920  coco/bbox_mAP_75: 0.4310  coco/bbox_mAP_s: 0.2400  coco/bbox_mAP_m: 0.4260  coco/bbox_mAP_l: 0.5410
# * cfg("deformable_detr_swin-s_coco", 4), with mixed precision 2680360 -> finished due tue early stopping with patience 5; Epoch(val) [8][1250/1250]    coco/bbox_mAP: 0.3770  coco/bbox_mAP_50: 0.5620  coco/bbox_mAP_75: 0.4080  coco/bbox_mAP_s: 0.2290  coco/bbox_mAP_m: 0.4100  coco/bbox_mAP_l: 0.4960


# cfg("codino_convnext-s_coco", 2), 2681827 -> would have run but didn´t find checkpoint
# cfg("codino_swin-s_coco", 2), 2681828-> would have run but didn´t find checkpoint
# cfg("ddq_convnext-s_coco", 2), 2681829 -> RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and Float for the source.
# cfg("ddq_swin-s_coco", 2), 2681830 -> RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and Float for the source.
# ? cfg("paa_convnext-s_coco", 2), 2681831 -> RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss.
# ? cfg("paa_swin-s_coco", 2), 2681832 -> RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss.
# * cfg("sparse_rcnn_convnext-s_coco", 2), 2681833 -> Epoch(val) [13][2500/2500]    coco/bbox_mAP: 0.4240  coco/bbox_mAP_50: 0.6130  coco/bbox_mAP_75: 0.4630  coco/bbox_mAP_s: 0.2560  coco/bbox_mAP_m: 0.4530  coco/bbox_mAP_l: 0.5790
# cfg("sparse_rcnn_swin-s_coco", 2), 2681834 -> would have run but didn´t find checkpoint
# ? cfg("tood_convnext-s_coco", 2), 2681835 -> RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss.
# ? cfg("tood_swin-s_coco", 2), 2681835 -> RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss.

# 2681876 - 2681880 + 2681883

#! cfg("codino_convnext-s_coco", 2), 2681883 -> runs
#! cfg("codino_swin-s_coco", 2), 2681876 -> runs
# ? cfg("ddq_convnext-s_coco", 2), 2681878 -> memory error after 1 epoch
#! cfg("ddq_swin-s_coco", 2), 2681878 -> runs
#! cfg("sparse_rcnn_swin-s_coco", 2), 2681883 -> runs
#! cfg("glip_convnext-s_coco", 2)
# * cfg("dino_convnext-s_coco", 2) -> memory error
# * cfg("dino_swin-s_coco", 2) 2682144 -> memory error
# * cfg("cascade_rcnn_convnext-s_coco", 2), ->
# * cfg("deformable_detr_convnext-s_coco", 4), ->
# * cfg("deformable_detr_swin-s_coco", 4), ->
# 2682140 till 2682145


# * cfg("dino_convnext-s_coco", 2) with_cp=true; 2682373 ->
# * cfg("dino_swin-s_coco", 2) with_cp=true ; 2682374 ->


#! cfg("rtmdet_convnext-s_coco", 2),  ->
#! cfg("rtmdet_swin-s_coco", 2),  ->
#! cfg("ddq_convnext-s_coco", 2),  ->
# * cfg("sparse_rcnn_convnext-s_coco", 2),  ->
# 2683254 - 2683257

#! cfg("glip_swin-s_coco", 2), 2683265 ->


# ? cfg("paa_convnext-s_coco", 2), 2683302 -> AttributeError: 'NoneType' object has no attribute 'split'Exception ignored on calling ctypes callback function: <function _ThreadpoolInfo._find_modules_with_dl_iterate_phdr.<locals>.match_module_callback at 0x14f663318ee0>
# ? cfg("paa_swin-s_coco", 2), 2683303 -> AttributeError: 'NoneType' object has no attribute 'split'Exception ignored on calling ctypes callback function: <function _ThreadpoolInfo._find_modules_with_dl_iterate_phdr.<locals>.match_module_callback at 0x14f663318ee0>
# ? cfg("tood_convnext-s_coco", 2), ->
# ? cfg("tood_swin-s_coco", 2), ->
# 2683302 - 2683305

# 2683341
# 2683342

# cfg("dino_swin-b_coco", 4), accelerated, 2669626_submission -> memory error
# cfg("dino_swin-b_coco", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device

# cfg("rtmdet_convnext-b_coco", 1), accelerated-h100, 2669839_submitted -> running till val loop than crashed because of CUDA error: no kernel image is available for execution on the device
# cfg("rtmdet_swin-b_coco", 3), accelerated-h100, 2669840_submission -> running till val loop than crashed because of CUDA error: no kernel image is available for execution on the device
# cfg("dino_convnext-b_coco", 2), accelerated-h100, 2669841_submission -> memory error and child process exited

# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in train.py v1, 2671230_submission -> RuntimeError: `scale_lr` should be called before building ParamScheduler because ParamScheduler will store initial lr from optimizer wrappers
# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in train.py v2, 2671256_submission -> RuntimeError: `scale_lr` should be called before building ParamScheduler because ParamScheduler will store initial lr from optimizer wrappers
# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in distributed.py, 2671264_submission ->     raise ValueError('There is no tensor in the inputs, 'ValueError: There is no tensor in the inputs, cannot get dtype and device.
# cfg("dino_swin-b_coco", 4), accelerated, with AvoidCUDAOOM in distributed.py with type to tensor changing, 2671284_submission ->     raise ValueError('There is no tensor in the inputs, 'ValueError: There is no tensor in the inputs, cannot get dtype and device.


# cfg("dino_swin-b_voc0712", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device
# cfg("dino_swin-b_voc0712", 4), accelerated-h100, 2669719_submission -> CUDA error: no kernel image is available for execution on the device


for cfg in cfgs:
    # Generate a random port number between 10000 and 60000
    port = random.randint(10000, 60000)

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
        # Set the PORT environment variable using slurm_additional_parameters
        slurm_additional_parameters={"export": f"ALL,PORT={port}"},
    )
    executor.submit(command)
    logger.info(f"Submitted job for {cfg} with PORT {port}")
