import subprocess
from new_tester import tester


def test_with_multiple_gpus(
    config_path,
    checkpoint,
    work_dir,
    gpus,
    nnodes=1,
    node_rank=0,
    port=29500,
    master_addr="127.0.0.1",
    out=None,
    show=False,
    show_dir=None,
    wait_time=2,
    cfg_options=None,
    launcher="none",
    tta=False,
    local_rank=0,
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
        tester(
            config_path=config_path,
            checkpoint=checkpoint,
            work_dir=work_dir,
            out=out,
            show=show,
            show_dir=show_dir,
            wait_time=wait_time,
            cfg_options=cfg_options,
            launcher=launcher,
            tta=tta,
            local_rank=local_rank,
        ),
    ]
    subprocess.run(command, check=True)
