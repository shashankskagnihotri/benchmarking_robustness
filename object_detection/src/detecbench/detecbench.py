""" """

import mmdet  # noqa
from dataclasses import asdict, dataclass
from typing import Literal
import tyro
from mmengine.runner import Runner
from attacks import PGD, FGSM, BIM, replace_val_loop
from corruptions import CommonCorruption, CommonCorruption3d
from mmengine.config import Config


def detecbench(
    task: PGD | FGSM | BIM | CommonCorruption | CommonCorruption3d,
    model: str,
    checkpoint: str,
    log_dir: str = "./logs",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
):
    """"""

    # Parse task type
    is_attack = False
    is_cc = False
    is_3dcc = False
    if isinstance(task, PGD):
        is_attack = True
        task_name = "PGD"
    elif isinstance(task, FGSM):
        is_attack = True
        task_name = "FGSM"
    elif isinstance(task, BIM):
        is_attack = True
        task_name = "BIM"
    elif isinstance(task, CommonCorruption):
        is_cc = True
        task_name = "CC"
    elif isinstance(task, CommonCorruption3d):
        is_3dcc = True
        task_name = "3DCC"
    else:
        raise NotImplementedError

    # Setup the configuration
    cfg = Config.fromfile(model)
    cfg.work_dir = log_dir
    cfg.load_from = checkpoint

    if is_cc:
        assert not isinstance(task, (PGD, FGSM, BIM))
        if task.dataset == "Pascal":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/severity_{task.severity}/"
        elif task.dataset == "Coco":
            cfg.val_dataloader.dataset.data_prefix.img = f"{task.name}/severity_{task.severity}/"
    elif is_3dcc:
        assert not isinstance(task, (PGD, FGSM, BIM))
        if task.dataset == "Pascal":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/{task.severity}/"
        elif task.dataset == "Coco":
            cfg.val_dataloader.dataset.data_prefix.img = f"{task.name}/{task.severity}/"

    # Setup optional wandb logging
    use_wandb = wandb_project is not None and wandb_entity is not None
    if use_wandb:
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.interval = 1000
        model_name = model.split("/")[-1].split(".")[0]
        cfg.visualizer.vis_backends = dict(
            dict(
                type="WandbVisBackend",
                init_kwargs={
                    "project": wandb_project,
                    "entity": wandb_entity,
                    "config": {
                        "task_name": task_name,
                        "model": "model",
                        "checkpoint": checkpoint,
                        **{
                            k: v for k, v in task.__dict__.items() if not k.startswith("_")
                        },  # attack attributes
                    },
                    "group": model_name,
                },
            )
        )

    # Initialize the runner
    runner = Runner.from_cfg(cfg)

    if is_attack:
        # Setup the modified validation loop
        assert not isinstance(task, (CommonCorruption, CommonCorruption3d))
        replace_val_loop(attack=task, use_wandb=use_wandb)

    # Run the attack
    runner.val()


if __name__ == "__main__":
    tyro.cli(detecbench)
