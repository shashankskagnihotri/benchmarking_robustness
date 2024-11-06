""" """

import mmdet  # noqa
from dataclasses import asdict, dataclass
from typing import Literal
import tyro
from mmengine.runner import Runner
from attacks import pgd_attack, fgsm_attack, bim_attack, replace_val_loop
from mmengine.config import Config


@dataclass
class EvalConfig:
    """"""

    # model: str = "src/detecbench/configs/yolox_tiny_8xb8-300e_coco.py"
    model: str
    # checkpoint: str = "src/detecbench/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
    checkpoint: str
    log_dir: str = "./logs"
    wandb_project: str | None = None
    wandb_entity: str | None = None


@dataclass
class Pgd:
    """PGD attack from ..."""

    # Number of iteration steps for PGD
    steps: int = 20
    # Maximum perturbation value
    epsilon: float = 8
    # Step size for gradient updates
    alpha: float = 0.01 * 255
    # Whether attack should be targeted
    targeted: bool = False
    # Whether to use random initialization
    random_start: bool = False

    # epsilon: float,
    # alpha: float,
    # norm: str,
    # targeted: bool,
    # steps: int,


@dataclass
class Bim:
    """"""

    steps: int = 20
    epsilon: float = 8
    alpha: float = 0.01 * 255
    targeted: bool = False


@dataclass
class Fgsm:
    """"""

    epsilon: float = 8
    alpha: float = 0.01 * 255
    targeted: bool = False  # TODO implement
    random_start: bool = False  # TODO implement


@dataclass
class CommonCorruption:
    """"""

    name: str = "cc_"
    severity: int = 3
    generate_missing: bool = False
    dataset: Literal["Coco", "Pascal"] = "Coco"


@dataclass
class CommonCorruption3d:
    """"""

    name: str = "3dcc_"
    severity: int = 3
    generate_missing: bool = False
    dataset: Literal["Coco", "Pascal"] = "Coco"


def main(config: EvalConfig, task: Pgd | Fgsm | Bim | CommonCorruption | CommonCorruption3d):
    # Parse task type
    is_attack = False
    is_cc = False
    is_3dcc = False
    if isinstance(task, Pgd):
        attack_function = pgd_attack
        is_attack = True
        task_name = "PGD"
    elif isinstance(task, Fgsm):
        attack_function = fgsm_attack
        is_attack = True
        task_name = "FGSM"
    elif isinstance(task, Bim):
        attack_function = bim_attack
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
    cfg = Config.fromfile(config.model)
    cfg.work_dir = config.log_dir
    cfg.load_from = config.checkpoint

    if is_cc:
        assert not isinstance(task, (Pgd, Fgsm, Bim))
        if task.dataset == "Pascal":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/severity_{task.severity}/"
        elif task.dataset == "Coco":
            cfg.val_dataloader.dataset.data_prefix.img = f"{task.name}/severity_{task.severity}/"
    elif is_3dcc:
        assert not isinstance(task, (Pgd, Fgsm, Bim))
        if task.dataset == "Pascal":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/{task.severity}/"
        elif task.dataset == "Coco":
            cfg.val_dataloader.dataset.data_prefix.img = f"{task.name}/{task.severity}/"

    # Setup optional wandb logging
    use_wandb = config.wandb_project is not None and config.wandb_entity is not None
    if use_wandb:
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.interval = 1000
        model_name = config.model.split("/")[-1].split(".")[0]
        cfg.visualizer.vis_backends = dict(
            dict(
                type="WandbVisBackend",
                init_kwargs={
                    "project": config.wandb_project,
                    "entity": config.wandb_entity,
                    "config": {
                        "task_name": task_name,
                        **asdict(config),
                        **asdict(task),
                    },
                    "group": model_name,
                },
            )
        )

    # Initialize the runner
    runner = Runner.from_cfg(cfg)

    if is_attack:
        # Setup the modified validation loop
        replace_val_loop(attack=attack_function, attack_kwargs=asdict(task), use_wandb=use_wandb)

    # Run the attack
    runner.val()


if __name__ == "__main__":
    tyro.cli(main)
