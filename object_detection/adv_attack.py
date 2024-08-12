import argparse
import logging
from time import sleep
from mmdet.structures.det_data_sample import DetDataSample
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from torchvision import transforms
from typing import Callable
from typing import Sequence
from dotenv import load_dotenv
import os
from typing import Dict, List, Union
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator
import wandb
from copy import deepcopy
import cospgd
import numpy as np
import torch.nn.functional as F

load_dotenv()
WAND_PROJECT = os.getenv("WANDB_PROJECT")
WAND_ENTITY = os.getenv("WANDB_ENTITY")
assert WAND_PROJECT, "Please set the WANDB_PROJECT environment variable"
assert WAND_ENTITY, "Please set the WANDB_ENTITY environment variable"


def cospgd_scale(
    predictions: torch.Tensor,
    loss: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    targeted: bool,
):
    """
    Scale the loss based on the pixel-wise predictions.

    Args:
        predictions (torch.Tensor): The pixel-wise predictions. Shape (batch_size, num_classes, height, width).
        loss (torch.Tensor): The loss tensor.
        labels (torch.Tensor): The target labels. Shape (batch_size, num_classes, height, width)
        num_classes (int): The number of classes.
        targeted (bool): Whether the attack is targeted or not.

    Returns:
        torch.Tensor: The scaled loss tensor.
    """
    cossim = torch.nn.functional.cosine_similarity(predictions, labels, dim=1)
    if targeted:
        cossim = 1 - cossim
    return cossim * loss


def pixel_wise_pred(image: torch.Tensor, bboxes, labels, num_classes: int, scores=None):
    _, img_h, img_w = image.shape

    result_image = torch.zeros((num_classes, img_h, img_w)).to("cuda")

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Deal with out of bound boxes
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w - 1, x_max)
        y_max = min(img_h - 1, y_max)

        # Draw the bounding box
        if scores is not None:
            result_image[label, y_min : y_max + 1, x_min : x_max + 1] = scores[i]
        else:
            result_image[label, y_min : y_max + 1, x_min : x_max + 1] = 1

    return result_image.reshape(1, num_classes, img_h, img_w)


def preprocess_data(data_batch: dict, runner):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)
    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    mean = runner.model.data_preprocessor.mean
    std = runner.model.data_preprocessor.std
    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")
    adv_images.requires_grad = True
    return data_batch_prepro, images, adv_images


def cospgd_attack(
    data_batch: dict,
    runner,
    steps: int,
    epsilon: float,
    alpha: float,
    targeted: bool,
    random_start: bool,
    evaluators: List[Evaluator],
):
    # Preprocess the data
    data_batch_prepro, images, adv_images = preprocess_data(data_batch, runner)

    # perform evaluation on the original image
    evaluator_process(evaluators[0], data_batch_prepro, runner)

    # setup for pixel-wise loss
    # runner.model.loss_cl.reduction = "none"
    # cfg.model.bbox_head.loss_bbox
    # cfg.model.bbox_head.loss_cls

    for step in range(steps):
        data_batch_prepro["inputs"][0] = adv_images  # update with latest adv image
        runner.model.training = True  # avoid missing arguments error in some models

        runner.model.zero_grad()
        adv_images.grad = None

        # norm == inf
        images = cospgd.functions.init_linf(
            images, epsilon=epsilon, clamp_min=0, clamp_max=255
        )

        # assumes batchsize=1
        assert len(data_batch_prepro["data_samples"]) == 1
        target_labels = data_batch_prepro["data_samples"][0].gt_instances.labels
        target_bboxes = data_batch_prepro["data_samples"][0].gt_instances.bboxes

        if targeted:
            target_labels = torch.ones_like(target_labels)

        target_pixelwise = pixel_wise_pred(
            adv_images[0],
            labels=target_labels,
            bboxes=target_bboxes,
            num_classes=runner.model.bbox_head.num_classes,
        )

        output = runner.model(**data_batch_prepro, mode="predict")

        pred_labels = output[0].pred_instances.labels
        pred_bboxes = output[0].pred_instances.bboxes
        pred_scores = output[0].pred_instances.scores

        pred_pixelwise = pixel_wise_pred(
            adv_images[0],
            labels=pred_labels,
            bboxes=pred_bboxes,
            num_classes=runner.model.bbox_head.num_classes,
            scores=pred_scores,
        )

        # TODO: get loss from output, avoid one forward pass
        # TODO: pixel-wise loss
        # losses = runner.model(**data_batch_prepro, mode="loss")
        # loss, _ = runner.model.parse_losses(losses)

        loss = F.cross_entropy(pred_pixelwise, target_pixelwise, reduction="none")

        assert isinstance(loss, torch.Tensor)

        scaled_loss = cospgd_scale(
            predictions=pred_pixelwise,
            loss=loss,
            labels=target_pixelwise,
            num_classes=runner.model.bbox_head.num_classes,
            targeted=targeted,
        )

        # assert scaled_loss.shape == adv_images[0].shape
        scaled_loss = scaled_loss.mean()
        scaled_loss.backward(retain_graph=True)

        adv_images = cospgd.functions.step_inf(
            perturbed_image=adv_images,
            orig_image=images,
            epsilon=epsilon,
            data_grad=adv_images.grad,
            alpha=alpha,
            targeted=targeted,
            clamp_max=255,
        )

        # TODO: log loss and scaled loss

        mean = runner.model.data_preprocessor.mean
        std = runner.model.data_preprocessor.std
        data_batch_prepro["inputs"][0] = transforms.Normalize(mean, std)(adv_images)
        runner.model.training = False  # avoid missing arguments error in some models
        evaluator_process(evaluators[step + 1], data_batch_prepro, runner)

    return data_batch_prepro


def pgd_attack(
    data_batch: dict,
    runner,
    steps: int,
    epsilon: float,
    alpha: float,
    targeted: bool,
    random_start: bool,
    evaluators: List[Evaluator],
):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    # Preprocess the data
    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    assert isinstance(images, torch.Tensor)
    mean = runner.model.data_preprocessor.mean
    std = runner.model.data_preprocessor.std
    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    if targeted:
        raise NotImplementedError

    if random_start:
        raise NotImplementedError

    evaluator_process(evaluators[0], data_batch_prepro, runner)

    for step in range(steps):
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        runner.model.training = True  # avoid missing arguments error in some models
        losses = runner.model(**data_batch_prepro, mode="loss")
        cost, _ = runner.model.parse_losses(losses)

        runner.model.zero_grad()
        adv_images.grad = None
        cost.backward(retain_graph=True)
        grad = adv_images.grad
        assert grad is not None

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

        data_batch_prepro["inputs"][0] = transforms.Normalize(mean, std)(adv_images)
        runner.model.training = False  # avoid missing arguments error in some models
        evaluator_process(evaluators[step + 1], data_batch_prepro, runner)

    return data_batch_prepro


def evaluator_process(evaluator, data_batch_prepro, runner):
    with torch.no_grad():
        outputs = runner.model(**data_batch_prepro, mode="predict")

    evaluator.process(data_samples=outputs, data_batch=data_batch_prepro)


def fgsm_attack(
    data_batch: dict,
    runner,
    epsilon: float,
    alpha: float,
    norm: str,
    targeted: bool,
    evaluators: List[Evaluator],
):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    assert isinstance(images, torch.Tensor)

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    evaluator_process(evaluators[0], data_batch_prepro, runner)

    # Get gradient
    adv_images.requires_grad = True
    data_batch_prepro["inputs"][0] = adv_images

    runner.model.training = True  # avoid missing arguments error in some models
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

    if norm == "inf":
        delta = torch.clamp(adv_images - images, min=-1 * epsilon, max=epsilon)
    elif norm == "two":
        delta = adv_images - images
        batch_size = images.shape[0]
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

    # Clipping to maintain [0, 255] range
    adv_images = torch.clamp(images + delta, 0, 255)

    # Return the perturbed image
    adv_images.requires_grad = False
    data_batch_prepro["inputs"][0] = transforms.Normalize(mean, std)(adv_images)
    runner.model.training = False  # avoid missing arguments error in some models
    evaluator_process(evaluators[1], data_batch_prepro, runner)

    return data_batch_prepro


def bim_attack(
    data_batch: dict,
    runner,
    epsilon: float,
    alpha: float,
    norm: str,
    targeted: bool,
    steps: int,
    evaluators: List[Evaluator],
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
    evaluator_process(evaluators[0], data_batch_prepro, runner)

    for step in range(steps):
        # Get gradient
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        runner.model.training = True  # avoid missing arguments error in some models
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

        data_batch_prepro["inputs"][0] = transforms.Normalize(mean, std)(adv_images)
        runner.model.training = False  # avoid missing arguments error in some models
        evaluator_process(evaluators[step + 1], data_batch_prepro, runner)

    return data_batch_prepro


# restores the tensors to their original scale
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def denorm(batch, mean, std):
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


def replace_val_loop(attack: Callable, attack_kwargs: dict):
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

        def run(self) -> list[dict]:
            self.runner.call_hook("before_val")
            self.runner.call_hook("before_val_epoch")
            self.runner.model.eval()

            steps = attack_kwargs.get("steps", 1)
            # +1 since we want to evaluate the original data as well
            evaluators = [deepcopy(self.evaluator) for _ in range(steps + 1)]

            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch, evaluators)

            metrics = [
                evaluator.evaluate(len(self.dataloader.dataset))  # type: ignore
                for evaluator in evaluators
            ]

            # Log history of metrics to wandb
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")

            for step, metric in enumerate(metrics):
                wandb.log({**metric, "step": step})
                sleep(2)

            self.runner.call_hook(
                "after_val_epoch", metrics=metrics[-1]
            )  # metrics now a list of dicts, standard hooks expect dict
            self.runner.call_hook("after_val")
            return metrics

        def run_iter(
            self,
            idx,
            data_batch: Sequence[dict],
            evaluators: List[Evaluator],
        ):
            self.runner.call_hook(
                "before_val_iter", batch_idx=idx, data_batch=data_batch
            )

            data_batch_prepro = attack(
                data_batch, self.runner, **attack_kwargs, evaluators=evaluators
            )

            with torch.no_grad():
                outputs = self.runner.model(**data_batch_prepro, mode="predict")

            self.runner.call_hook(
                "after_val_iter",
                batch_idx=idx,
                data_batch=data_batch_prepro,
                outputs=outputs,
            )


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
                "tags": ["debug"],
            },
        )
    )

    if "cospgd" in attack_name:
        cfg.model.bbox_head.loss_bbox.reduction = "none"
        cfg.model.bbox_head.loss_cls.reduction = "none"

    # Register the attack loop if an attack is provided
    if attack is not None:
        replace_val_loop(attack, attack_kwargs)

    # Initialize the runner
    runner = Runner.from_cfg(cfg)

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
        help="Directory path where result files are saved (default: ./work_dirs/logs)",
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
        attack = cospgd_attack
        attack_kwargs = {
            "steps": args.steps,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "targeted": args.targeted,
            "random_start": args.random_start,
        }
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
