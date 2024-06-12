import subprocess
from new_trainer import trainer


def train_with_multiple_gpus(
    config,
    work_dir,
    gpus,
    nnodes=1,
    node_rank=0,
    port=29500,
    master_addr="127.0.0.1",
):
    command = [
        "python",
        "-m",
        "torch.distributed.launch",
        "--nnodes=" + str(nnodes),
        "--node_rank=" + str(node_rank),
        "--master_addr=" + master_addr,
        "--nproc_per_node=" + str(gpus),
        "--master_port=" + str(port),
        trainer(config=config, work_dir=work_dir),
        "--launcher",
        "pytorch",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    train_with_multiple_gpus(
        config="configs_erroneous/verification/yolox_r101_coco.py",
        work_dir="slurm/work_dir/0_verification_submitit_verifier_trainer_tester/trainer",
        gpus=3,
    )

# yolox_r101_coco.py

# python -m pudb new_distributed_trainer.py
