import argparse
import json
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from typing import Dict, List, Optional, Sequence, Union
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader
from torchvision import transforms
from mmengine.evaluator import Evaluator
import logging
from typing import Callable
import shutil
import os
import collect_attack_results

DATA_BATCH = Optional[Union[dict, tuple, list]]


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
    images_clone = images.clone().float().detach()
    assert isinstance(images, torch.Tensor)

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    if targeted:
        raise NotImplementedError

    if random_start:
        raise NotImplementedError

    for _ in range(steps):
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        losses = runner.model(**data_batch_prepro, mode="loss")
        cost, _ = runner.model.parse_losses(losses)

        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )
        grad = torch.cat(grad, dim=0)

        adv_images = adv_images.detach() + alpha * grad.sign()
        # assert torch.allclose(ALPHA * grad.sign(), torch.zeros_like(grad.sign()))
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        # assert torch.allclose(delta, torch.zeros_like(delta))
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

    adv_images.requires_grad = False
    transforms.Normalize(mean, std, inplace=True)(adv_images)
    data_batch_prepro["inputs"][0] = adv_images

    if alpha == 0 or epsilon == 0 or steps == 0:  # for debug
        assert torch.allclose(images_clone, adv_images)
    else:
        assert not torch.allclose(images_clone, adv_images)

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
    grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)
    grad = torch.cat(grad, dim=0)

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
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )
        grad = torch.cat(grad, dim=0)

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


def run_attack_val(
    attack: Callable,
    config_file: str,
    checkpoint_file: str,
    attack_kwargs: dict,
    log_dir: str,
):
    LOOPS.module_dict.pop("ValLoop")

    @LOOPS.register_module()
    class ValLoop(BaseLoop):
        def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            evaluator: Union[Evaluator, Dict, List],
            fp16: bool = False,
        ) -> None:
            super().__init__(runner, dataloader)

            if isinstance(evaluator, (dict, list)):
                self.evaluator = runner.build_evaluator(evaluator)
            else:
                assert isinstance(evaluator, Evaluator), (
                    "evaluator must be one of dict, list or Evaluator instance, "
                    f"but got {type(evaluator)}."
                )
                self.evaluator = evaluator
            if hasattr(self.dataloader.dataset, "metainfo"):
                self.evaluator.dataset_meta = getattr(
                    self.dataloader.dataset, "metainfo"
                )
                self.runner.visualizer.dataset_meta = getattr(
                    self.dataloader.dataset, "metainfo"
                )
            else:
                print_log(
                    f"Dataset {self.dataloader.dataset.__class__.__name__} has no "
                    "metainfo. ``dataset_meta`` in evaluator, metric and "
                    "visualizer will be None.",
                    logger="current",
                    level=logging.WARNING,
                )
            self.fp16 = fp16

        def run(self) -> dict:
            self.runner.call_hook("before_val")
            self.runner.call_hook("before_val_epoch")
            self.runner.model.eval()
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)

            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))  # type: ignore
            self.runner.call_hook("after_val_epoch", metrics=metrics)
            self.runner.call_hook("after_val")
            return metrics

        def run_iter(
            self,
            idx,
            data_batch: Sequence[dict],
        ):
            self.runner.call_hook(
                "before_val_iter", batch_idx=idx, data_batch=data_batch
            )

            data_batch_prepro = attack(data_batch, self.runner, **attack_kwargs)

            with torch.no_grad():
                outputs = self.runner.model(**data_batch_prepro, mode="predict")

            self.evaluator.process(data_samples=outputs, data_batch=data_batch_prepro)
            self.runner.call_hook(
                "after_val_iter",
                batch_idx=idx,
                data_batch=data_batch_prepro,
                outputs=outputs,
            )

    cfg = Config.fromfile(config_file)
    cfg.work_dir = log_dir
    cfg.load_from = checkpoint_file
    cfg.checkpoint_config = dict(interval=0)
    cfg.default_hooks.logger.interval = 100

    runner = Runner.from_cfg(cfg)
    runner.val()

    # save cfg as json
    destination_file = os.path.join(runner.work_dir, "cfg.json")
    cfg.dump(destination_file)

    # copy metrics json into right folder
    source_file = os.path.join(runner.work_dir, "vis_data", "scalars.json")
    destination_folder = runner.work_dir
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    destination_file = os.path.join(destination_folder, "metrics.json")
    shutil.copy(source_file, destination_file)

    # save kwargs as json
    destination_file = os.path.join(runner.work_dir, "args.json")
    attack_kwargs["attack"] = attack.__name__
    print("\nattack_kwargs: ", attack_kwargs)
    with open(destination_file, "w") as json_file:
        json.dump(attack_kwargs, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for the variables")
    parser.add_argument(
        "--targeted", action="store_true", help="Enable targeted attack"
    )
    parser.add_argument(
        "--random_start", action="store_true", help="Enable random start for attack"
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of steps for the attack"
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
    parser.add_argument(
        "--collect_results", action="store_true", help="Collect attack results"
    )

    args = parser.parse_args()

    targeted = args.targeted
    random_start = args.random_start
    steps = args.steps
    alpha = args.alpha
    epsilon = args.epsilon
    attack = args.attack
    checkpoint_file = args.checkpoint_file
    config_file = args.config_file
    norm = args.norm
    output_dir = args.output_dir
    collect_results = args.collect_results

    # Select right attack function
    if attack == "pgd":
        attack = pgd_attack
        attack_kwargs = {
            "steps": steps,
            "epsilon": epsilon,
            "alpha": alpha,
            "targeted": targeted,
            "random_start": random_start,
        }
    elif attack == "cospgd":
        attack_kwargs = {}
        raise NotImplementedError
    elif attack == "fgsm":
        attack = fgsm_attack
        attack_kwargs = {
            "epsilon": epsilon,
            "alpha": alpha,
            "targeted": targeted,
            "norm": norm,
        }
    elif attack == "bim":
        attack = bim_attack
        attack_kwargs = {
            "epsilon": epsilon,
            "alpha": alpha,
            "targeted": targeted,
            "norm": norm,
            "steps": steps,
        }
    else:
        raise ValueError

    run_attack_val(
        attack=attack,
        attack_kwargs=attack_kwargs,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        log_dir=output_dir,
    )

    if collect_results:
        collect_attack_results.collect_results()
