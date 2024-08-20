import argparse
import os

from dotenv import load_dotenv
from mmengine.config import Config
from mmengine.runner import Runner

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
                "group": model_name,
            },
        )
    )

    if "3dcc" in corruption_name and "voc" in model_name:
        cfg.val_dataloader.dataset.img_subdir = f"{corruption_name}/{severity}/"
    elif "3dcc" in corruption_name:  # coco
        cfg.val_dataloader.dataset.data_prefix.img = f"{corruption_name}/{severity}/"
    elif "cc" in corruption_name and "voc" in model_name:
        cfg.val_dataloader.dataset.img_subdir = (
            f"{corruption_name}/severity_{severity}/"
        )
    elif "cc" in corruption_name:  # coco
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
        default="models_tmp/atss_r50_voc0712/atss_r50_voc0712.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="models_tmp/atss_r50_voc0712/epoch_4.pth",
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
