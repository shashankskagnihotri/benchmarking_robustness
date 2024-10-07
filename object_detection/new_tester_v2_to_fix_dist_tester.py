import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    args = parser.parse_args()
    return args


def tester(
    config_path,
    checkpoint,
    work_dir=None,
    out=None,
    show=False,
    show_dir=None,
    wait_time=2,
    cfg_options=None,
    launcher="none",
    tta=False,
    local_rank=0,
):
    # Generate the command for testing
    command = [
        "python",
        osp.realpath(__file__),  # path to the current script
        config_path,
        checkpoint,
        "--work-dir",
        work_dir or "",  # Use provided work_dir or an empty string
    ]
    if out:
        command.extend(["--out", out])
    if show:
        command.extend(["--show"])
    if show_dir:
        command.extend(["--show-dir", show_dir])
    if cfg_options:
        for key, value in cfg_options.items():
            command.extend(["--cfg-options", f"{key}={value}"])
    if tta:
        command.extend(["--tta"])

    return command


def main():
    args = parse_args()
    tester(
        args.config,
        args.checkpoint,
        args.work_dir,
        out=None,
        show=False,
        show_dir=None,
        wait_time=2,
        cfg_options=None,
        launcher="none",
        tta=False,
        local_rank=0,
    )


if __name__ == "__main__":
    main()
