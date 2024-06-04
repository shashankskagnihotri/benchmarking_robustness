import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.runner import Hook
from mmdet.engine.hooks.visualization_hook import DetVisualizationHook
from mmengine.visualization import Visualizer
from mmengine.fileio import get
from mmdet.structures import DetDataSample, TrackDataSample
import mmcv
from torchvision import transforms
from typing import Callable
import wandb
from typing import Sequence
from dotenv import load_dotenv
import os

load_dotenv()
WAND_PROJECT = os.getenv("WANDB_PROJECT")
WAND_ENTITY = os.getenv("WANDB_ENTITY")
assert WAND_PROJECT, "Please set the WANDB_PROJECT environment variable"
assert WAND_ENTITY, "Please set the WANDB_ENTITY environment variable"


def pgd_attack(
    data_batch: dict,
    runner,
    steps: int,
    epsilon: float,
    alpha: float,
    targeted: bool,
    random_start: bool,
):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")
    adv_images.requires_grad = True

    if targeted:
        raise NotImplementedError

    if random_start:
        raise NotImplementedError

    for _ in range(steps):
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        losses = runner.model(**data_batch_prepro, mode="loss")
        cost, _ = runner.model.parse_losses(losses)

        runner.model.zero_grad()
        cost.backward(retain_graph=True)
        grad = adv_images.grad
        assert grad is not None

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

    adv_images.requires_grad = False
    transforms.Normalize(mean, std, inplace=True)(adv_images)
    data_batch_prepro["inputs"][0] = adv_images

    return data_batch_prepro


def fgsm_attack(
    data_batch: dict, runner, epsilon: float, alpha: float, norm: str, targeted: bool
):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    assert isinstance(images, torch.Tensor)

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    # Get gradient
    adv_images.requires_grad = True
    data_batch_prepro["inputs"][0] = adv_images
    losses = runner.model(**data_batch_prepro, mode="loss")
    cost, _ = runner.model.parse_losses(losses)
    runner.model.zero_grad()
    cost.backward(retain_graph=True)
    grad = adv_images.grad
    assert grad is not None

    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    if targeted:
        sign_data_grad *= -1
    adv_images = adv_images.detach() + alpha * sign_data_grad

    # Adding clipping to maintain [0,1] range
    if norm == "inf":
        delta = torch.clamp(adv_images - images, min=-1 * epsilon, max=epsilon)
    elif norm == "two":
        delta = adv_images - images
        batch_size = images.shape[0]
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
    adv_images = torch.clamp(images + delta, 0, 255)

    # Return the perturbed image
    adv_images.requires_grad = False
    transforms.Normalize(mean, std, inplace=True)(adv_images)
    data_batch_prepro["inputs"][0] = adv_images

    return data_batch_prepro


def bim_attack(
    data_batch: dict,
    runner,
    epsilon: float,
    alpha: float,
    norm: str,
    targeted: bool,
    steps: int,
):
    """see https://arxiv.org/pdf/1607.02533.pdf"""
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    assert isinstance(images, torch.Tensor)

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    for _ in range(steps):
        # Get gradient
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        losses = runner.model(**data_batch_prepro, mode="loss")
        cost, _ = runner.model.parse_losses(losses)
        runner.model.zero_grad()
        cost.backward(retain_graph=True)
        grad = adv_images.grad
        assert grad is not None

        # Collect the element-wise sign of the data gradient
        sign_data_grad = grad.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        if targeted:
            sign_data_grad *= -1
        adv_images = adv_images.detach() + alpha * sign_data_grad

        # Adding clipping to maintain [0,255] range
        if norm == "inf":
            delta = torch.clamp(adv_images - images, min=-1 * epsilon, max=epsilon)
        elif norm == "two":
            delta = adv_images - images
            batch_size = images.shape[0]
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
        adv_images = torch.clamp(images + delta, 0, 255)

    # Return the perturbed image
    adv_images.requires_grad = False
    transforms.Normalize(mean, std, inplace=True)(adv_images)
    data_batch_prepro["inputs"][0] = adv_images

    return data_batch_prepro


# restores the tensors to their original scale
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to("cuda")
    if isinstance(std, list):
        std = torch.tensor(std).to("cuda")

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


class AdversarialAttackHook(Hook):
    def __init__(self, attack: Callable, attack_kwargs: dict):
        self.attack = attack
        self.attack_kwargs = attack_kwargs

    def before_val_iter(self, runner, batch_idx: int, data_batch: Sequence[dict]):
        with torch.enable_grad():
            data_batch_prepro = self.attack(data_batch, runner, **self.attack_kwargs)
        runner.data_batch = data_batch_prepro  # overwrite the data_batch

    def after_val_iter(
        self, runner, batch_idx: int, data_batch: Sequence[dict], outputs
    ):
        pass


def run_attack_val(
    attack: Callable | None,
    config_file: str,
    checkpoint_file: str,
    attack_kwargs: dict,
    log_dir: str,
):
    # Setup the configuration
    cfg = Config.fromfile(config_file)
    cfg.work_dir = log_dir
    cfg.load_from = checkpoint_file
    cfg.default_hooks.visualization.draw = True
    cfg.default_hooks.visualization.interval = 1000
    model_name = config_file.split("/")[-1].split(".")[0]
    attack_name = attack.__name__ if attack is not None else "none"
    cfg.visualizer.vis_backends = dict(
        dict(
            type="WandbVisBackend",
            init_kwargs={
                "project": WAND_PROJECT,
                "entity": WAND_ENTITY,
                "config": {
                    "attack": attack_name,
                    "attack_kwargs": attack_kwargs,
                    "config_file": config_file,
                    "checkpoint_file": checkpoint_file,
                },
                "name": model_name,
                "group": model_name,
            },
        )
    )

    # Initialize the runner
    runner = Runner.from_cfg(cfg)

    # Register the attack hook if an attack is provided
    if attack is not None:
        runner.register_hook(AdversarialAttackHook(attack, attack_kwargs))

    # Run the attack
    runner.val()


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for the variables")
    parser.add_argument(
        "--targeted", action="store_true", help="Enable targeted attack"
    )
    parser.add_argument(
        "--random_start", action="store_true", help="Enable random start for attack"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of steps for the attack"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01 * 255, help="Alpha value for the attack"
    )
    parser.add_argument(
        "--epsilon", type=float, default=8, help="Epsilon value for the attack"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="pgd",
        choices=["pgd", "fgsm", "bim", "cospgd", "none"],
        help="Type of attack (default: pgd)",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="mmdetection/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="inf",
        choices=["inf", "two"],
        help="Norm for the attack (default: inf)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./work_dirs/",
        help="Directory path where result files are saved (default: ./slurm/logs)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Select right attack function
    if args.attack == "pgd":
        attack = pgd_attack
        attack_kwargs = {
            "steps": args.steps,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "targeted": args.targeted,
            "random_start": args.random_start,
        }
    elif args.attack == "cospgd":
        attack_kwargs = {}
        raise NotImplementedError
    elif args.attack == "fgsm":
        attack = fgsm_attack
        attack_kwargs = {
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "targeted": args.targeted,
            "norm": args.norm,
        }
    elif args.attack == "bim":
        attack = bim_attack
        attack_kwargs = {
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "targeted": args.targeted,
            "norm": args.norm,
            "steps": args.steps,
        }
    elif args.attack == "none":
        attack = None
        attack_kwargs = {}
    else:
        raise NotImplementedError

    run_attack_val(
        attack=attack,
        attack_kwargs=attack_kwargs,
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        log_dir=args.output_dir,
    )
