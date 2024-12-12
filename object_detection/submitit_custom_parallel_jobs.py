from dataclasses import dataclass
import random
import submitit
import numpy
import os

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
    # cfg("cascade_rcnn_convnext-s_coco", 2), #* with patience 15 Epoch(val) [22][2500/2500]    coco/bbox_mAP: 0.4790  coco/bbox_mAP_50: 0.6770  coco/bbox_mAP_75: 0.5230  coco/bbox_mAP_s: 0.3150  coco/bbox_mAP_m: 0.5190  coco/bbox_mAP_l: 0.6270
    # cfg("cascade_rcnn_swin-s_coco", 2), #* with patience 5 Epoch(val) [6][2500/2500]    coco/bbox_mAP: 0.4490  coco/bbox_mAP_50: 0.6500  coco/bbox_mAP_75: 0.4870  coco/bbox_mAP_s: 0.2710  coco/bbox_mAP_m: 0.4830  coco/bbox_mAP_l: 0.6060
    # cfg(
    #     "codino_convnext-s_coco", 2
    # ),  #! resumed epoch: 8, iter: 295720 ran till Epoch(train) [12][ 8200/29572]) resumed epoch: 11, iter: 384436 ran till Epoch(train) [15][ 8000/29572], resumed epoch: 14, iter: 473152 ran till Epoch(train) [18][ 6600/29572]
    cfg(
        "codino_swin-s_coco", 2
    ),  #! resumed epoch: 8, iter: 295720 ran till [11][24300/29572]), resumed epoch: 10, iter: 354864 ran till Epoch(train) [13][24100/29572]; resumed epoch: 12, iter: 414008 ran till Epoch(train) [15][23450/29572] eta: 13 days
    cfg(
        "ddq_convnext-s_coco", 2
    ),  #! resumed epoch: 10, iter: 295720 ran till Epoch(train) [15][ 9650/29572]) resumed epoch: 14, iter: 414008ran till Epoch(train) [19][ 6300/29572]); resumed epoch: 18, iter: 532296 ran till Epoch(train) [23][ 5200/29572] eta: 6 days
    cfg(
        "ddq_swin-s_coco", 2
    ),  #! resumed epoch: 6, iter: 177432 ran till Epoch(train) [11][  550/29572]); resumed epoch: 10, iter: 295720 ran till Epoch(train) [14][29550/29572]; resumed epoch: 14, iter: 414008 ran till Epoch(train) [18][28850/29572] eta: 8 days
    # cfg("deformable_detr_convnext-s_coco", 4), #* with patience 15 Epoch(val) [24][1250/1250]    coco/bbox_mAP: 0.4620  coco/bbox_mAP_50: 0.6550  coco/bbox_mAP_75: 0.5050  coco/bbox_mAP_s: 0.2890  coco/bbox_mAP_m: 0.4920  coco/bbox_mAP_l: 0.6140
    # cfg("deformable_detr_swin-s_coco", 4),  #* ran completly Epoch(val) [36][1250/1250]    coco/bbox_mAP: 0.4830  coco/bbox_mAP_50: 0.6700  coco/bbox_mAP_75: 0.5280  coco/bbox_mAP_s: 0.3250  coco/bbox_mAP_m: 0.5090  coco/bbox_mAP_l: 0.6250
    cfg(
        "dino_convnext-s_coco", 2
    ),  #! resubmit (resumed epoch: 20, iter: 709728 ran till Epoch(train) [26][ 7000/29572]); resumed epoch: 25, iter: 857588 ran till Epoch(train) [31][ 6700/29572], resumed epoch: 30, iter: 1005448 ran till Epoch(train) [36][ 6300/29572] eta -1 day
    cfg(
        "dino_swin-s_coco", 2
    ),  #! resubmit (resumed epoch: 19, iter: 680156 ran till Epoch(train) [24][24950/29572]); resumed epoch: 23, iter: 798444 ran till Epoch(train) [28][22950/29572], resumed epoch: 27, iter: 916732 ran till  Epoch(train) [32][22450/29572] eta: 2:22:59
    cfg(
        "glip_convnext-s_coco", 2
    ),  #! resubmit (resumed epoch: 6, iter: 351798 ran till Epoch(train)  [9][46250/58633]); resumed epoch: 8, iter: 469064 ran till Epoch(train) [11][46850/58633], resumed epoch: 10, iter: 586330 ran till Epoch(train) [13][44600/58633] eta: 16 days
    cfg(
        "glip_swin-s_coco", 2
    ),  #! resubmit (resumed epoch: 6, iter: 351798 ran till Epoch(train)  [9][32400/58633]; resumed epoch: 8, iter: 469064 ran till Epoch(train) [11][35100/58633] eta: 19 days, resumed epoch: 10, iter: 586330 ran till Epoch(train) [13][32400/58633] eta: 18 days
    # cfg(
    #     "paa_convnext-s_coco", 2
    # ),  #* stopped with patience 5, Epoch(val) [11][2500/2500]    coco/bbox_mAP: 0.4440  coco/bbox_mAP_50: 0.6340  coco/bbox_mAP_75: 0.4870  coco/bbox_mAP_s: 0.2880  coco/bbox_mAP_m: 0.4890  coco/bbox_mAP_l: 0.5930
    # cfg("paa_swin-s_coco", 2), #* with patience 5 Epoch(val) [11][2500/2500]    coco/bbox_mAP: 0.4500  coco/bbox_mAP_50: 0.6410  coco/bbox_mAP_75: 0.4900  coco/bbox_mAP_s: 0.2790  coco/bbox_mAP_m: 0.4940  coco/bbox_mAP_l: 0.6040
    # cfg(
    #     "rtmdet_convnext-s_coco", 2
    # ),  #! with patience 5: Epoch(val) [16][500/500]    coco/bbox_mAP: 0.3690  coco/bbox_mAP_50: 0.5440 coco/bbox_mAP_75: 0.3960  coco/bbox_mAP_s: 0.1780  coco/bbox_mAP_m: 0.4170  coco/bbox_mAP_l: 0.5540 (resumed epoch: 10, iter: 293170 ran till Epoch(train) [20][23800/29317]), resumed epoch: 19, iter: 557023 ran till Epoch(train) [29][22400/29317] eta: 1 day, resumed epoch: 28, iter: 820876 ran fully Epoch(val) [36][500/500]    coco/bbox_mAP: 0.4150  coco/bbox_mAP_50: 0.5960  coco/bbox_mAP_75: 0.4470  coco/bbox_mAP_s: 0.2220  coco/bbox_mAP_m: 0.4680  coco/bbox_mAP_l: 0.5990
    cfg(
        "rtmdet_swin-s_coco", 2
    ),  #! resubmit Epoch(val) [8][500/500]    coco/bbox_mAP: 0.2710  coco/bbox_mAP_50: 0.4370  coco/bbox_mAP_75: 0.2880  coco/bbox_mAP_s: 0.0880  coco/bbox_mAP_m: 0.3000  coco/bbox_mAP_l: 0.4430  data_time: 0.0069  time: 0.4472 (had no start epoch since it first was set to only save after 10 epoch which the first run did not hit (fixed since then), ran till Epoch(train)  [8][29300/29317]), resumed epoch: 8, iter: 234536 ran till Epoch(train) [18][ 6200/29317] eta: 3 days, resumed epoch: 17, iter: 498389 ran till Epoch(train) [27][ 2800/29317] eta: 2 days
    # cfg("sparse_rcnn_convnext-s_coco", 2),  #* ran full 36 epochs Epoch(val) [36][2500/2500]    coco/bbox_mAP: 0.4740  coco/bbox_mAP_50: 0.6700  coco/bbox_mAP_75: 0.5200  coco/bbox_mAP_s: 0.3030  coco/bbox_mAP_m: 0.5050  coco/bbox_mAP_l: 0.6430
    # cfg("sparse_rcnn_swin-s_coco", 2),  #* ran fully Epoch(val) [36][2500/2500]    coco/bbox_mAP: 0.4750  coco/bbox_mAP_50: 0.6710  coco/bbox_mAP_75: 0.5170  coco/bbox_mAP_s: 0.3040  coco/bbox_mAP_m: 0.5030  coco/bbox_mAP_l: 0.6420
    # cfg(
    #     "tood_convnext-s_coco", 2
    # ),  #* with patience 5 Epoch(val) [10][2500/2500]    coco/bbox_mAP: 0.4550  coco/bbox_mAP_50: 0.6390  coco/bbox_mAP_75: 0.4950  coco/bbox_mAP_s: 0.2910  coco/bbox_mAP_m: 0.5030  coco/bbox_mAP_l: 0.5860; with patience 15 Epoch(val) [26][2500/2500]    coco/bbox_mAP: 0.4790  coco/bbox_mAP_50: 0.6630  coco/bbox_mAP_75: 0.5220  coco/bbox_mAP_s: 0.3200  coco/bbox_mAP_m: 0.5180  coco/bbox_mAP_l: 0.6120
    # cfg(
    #     "tood_swin-s_coco", 2
    # ),  #* with patience 5 Epoch(val) [10][2500/2500]    coco/bbox_mAP: 0.4510  coco/bbox_mAP_50: 0.6360  coco/bbox_mAP_75: 0.4890  coco/bbox_mAP_s: 0.2840  coco/bbox_mAP_m: 0.4890  coco/bbox_mAP_l: 0.5940; with patience 15 Epoch(val) [26][2500/2500]    coco/bbox_mAP: 0.4590  coco/bbox_mAP_50: 0.6410  coco/bbox_mAP_75: 0.4970  coco/bbox_mAP_s: 0.3080  coco/bbox_mAP_m: 0.5030  coco/bbox_mAP_l: 0.5910
    # cfg(
    #     "tood_r101-dconv-c3-c5_fpn_ms-2x_coco", 2
    # ),  #* resubmit started for first time ran till Epoch(train) [13][11350/29317], resumed epoch: 12, iter: 351804 ran til Epoch(val) [24][2500/2500]    coco/bbox_mAP: 0.4900  coco/bbox_mAP_50: 0.6640  coco/bbox_mAP_75: 0.5320  coco/bbox_mAP_s: 0.3150  coco/bbox_mAP_m: 0.5290
]


executor = submitit.AutoExecutor(folder="slurm/train_results")

jobs = []


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


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
        slurm_time="2-00:00:00",  # "2-00:00:00" "00:30:00"
        slurm_partition="gpu_4_a100",  # accelerated for horeka gpu_4_a100 "dev_gpu_4_a100"
        slurm_gres=f"gpu:{cfg.num_gpus}",
        slurm_mail_user="ruben.weber@students.uni-mannheim.de",
        slurm_mail_type="END,FAIL",
        # Set the PORT environment variable using slurm_additional_parameters
        slurm_additional_parameters={"export": f"ALL,PORT={port}"},
    )
    executor.submit(command)
    logger.info(f"Submitted job for {cfg} with PORT {port}")
