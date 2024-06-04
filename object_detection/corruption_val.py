import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from dotenv import load_dotenv
import os


# move this?
load_dotenv()
WAND_PROJECT = os.getenv("WANDB_PROJECT")
WAND_ENTITY = os.getenv("WANDB_ENTITY")
assert WAND_PROJECT, "Please set the WANDB_PROJECT environment variable"
assert WAND_ENTITY, "Please set the WANDB_ENTITY environment variable"


def run_corruption_val(
    corruption_name: str,
    severity: int,
    config_file: str,
    checkpoint_file: str,
    log_dir: str,
):
    # Setup the configuration
    cfg = Config.fromfile(config_file)
    cfg.work_dir = log_dir
    cfg.load_from = checkpoint_file
    cfg.default_hooks.visualization.draw = True
    cfg.default_hooks.visualization.interval = 500
    model_name = config_file.split("/")[-1].split(".")[0]
    cfg.visualizer.vis_backends = dict(
        dict(
            type="WandbVisBackend",
            init_kwargs={
                "project": "corruptions",
                "entity": WAND_ENTITY,
                "config": {
                    "corruption": corruption_name,
                    "severity": severity,
                    "config_file": config_file,
                    "checkpoint_file": checkpoint_file,
                },
                "name": model_name,
                "group": model_name,
            },
        )
    )
    if "3dcc" in corruption_name:
        cfg.val_dataloader.dataset.data_prefix.img = f"{corruption_name}/{severity}/"
    elif "cc" in corruption_name:
        cfg.val_dataloader.dataset.data_prefix.img = (
            f"{corruption_name}/severity_{severity}/"
        )
    elif "none" == corruption_name:
        pass
    else:
        raise ValueError(f"Unknown corruption name: {corruption_name}")

    # Initialize the runner
    runner = Runner.from_cfg(cfg)

    # Run the attack
    runner.val()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corruption_name", type=str)
    parser.add_argument("severity", type=int)
    parser.add_argument(
        "--config_file",
        type=str,
        default="mmdetection/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./work_dirs/",
        help="Directory path where result files are saved (default: ./work_dirs/logs)",
    )

    args = parser.parse_args()

    run_corruption_val(
        args.corruption_name,
        args.severity,
        args.config_file,
        args.checkpoint_file,
        args.log_dir,
    )
