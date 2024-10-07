from new_tester import tester
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    test_command = tester(
        config_path=config_path,
        checkpoint=checkpoint,
        work_dir=work_dir,
    )
    logger.debug(f"Tester returned: {test_command}")

    command = [
        "python",
        "-m",
        "torch.distributed.launch",
        "--nnodes=" + str(nnodes),
        "--node_rank=" + str(node_rank),
        "--master_addr=" + master_addr,
        "--nproc_per_node=" + str(gpus),
        "--master_port=" + str(port),
        test_command,
    ]
    logger.debug(f"Whole Command: {command}")

    subprocess.run(command, check=True)
