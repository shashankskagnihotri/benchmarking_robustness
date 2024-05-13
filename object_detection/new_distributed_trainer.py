import subprocess
from new_trainer import trainer  # Assuming train function is defined in new_trainer.py


def train_with_multiple_gpus(
    config,
    work_dir,
    gpus,
    auto_scale_lr,
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
        trainer(config=config, work_dir=work_dir, auto_scale_lr=auto_scale_lr),
        "--launcher",
        "pytorch",
    ]
    subprocess.run(command, check=True)


# Example usage:
# train_with_multiple_gpus("config_file.py", gpus=4)
