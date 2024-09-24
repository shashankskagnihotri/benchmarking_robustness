import argparse
import logging
import os
from copy import deepcopy
from time import sleep
from typing import Callable, Dict, List, Sequence, Union

import cospgd
import torch
import torch.nn.functional as F
import torch.version
from dotenv import load_dotenv
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner import Runner
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb

load_dotenv()
WAND_PROJECT = os.getenv("WANDB_PROJECT")
WAND_ENTITY = os.getenv("WANDB_ENTITY")
assert WAND_PROJECT, "Please set the WANDB_PROJECT environment variable"
assert WAND_ENTITY, "Please set the WANDB_ENTITY environment variable"

DEBUG = True

if DEBUG:
    WAND_PROJECT = "debug"

count_early_stopping = 0
image_counter = 0


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
        predictions (torch.Tensor): The pixel-wise predictions. Shape (num_classes, height, width).
        loss (torch.Tensor): The loss tensor.
        labels (torch.Tensor): The target labels. Shape (num_classes, height, width)
        num_classes (int): The number of classes.
        targeted (bool): Whether the attack is targeted or not.

    Returns:
        torch.Tensor: The scaled loss tensor.
    """
    cossim = F.cosine_similarity(
        F.softmax(predictions, dim=0), F.softmax(labels, dim=0), dim=0
    )

    if targeted:
        cossim = 1 - cossim
    return cossim * loss


@torch.enable_grad()
def pixel_wise_pred(data_batch_prepro, model, adv_images):
    with torch.enable_grad():
        output = model.test_step(data_batch_prepro)

        labels = output[0].pred_instances.labels
        bboxes = output[0].pred_instances.bboxes
        scores = output[0].pred_instances.scores

        # if len(labels) == 0:
        #     return None

        _, img_h, img_w = data_batch_prepro.get("inputs")[0].shape

        num_classes = 80
        result = torch.zeros((num_classes, img_h, img_w), device="cuda")

        # Clip bboxes to image dimensions
        x_min = torch.clamp(bboxes[:, 0], 0, img_w - 1).long()
        y_min = torch.clamp(bboxes[:, 1], 0, img_h - 1).long()
        x_max = torch.clamp(bboxes[:, 2], 0, img_w - 1).long()
        y_max = torch.clamp(bboxes[:, 3], 0, img_h - 1).long()

        # Vectorized bbox filling
        for i in range(bboxes.size(0)):
            result[labels[i], y_min[i] : y_max[i] + 1, x_min[i] : x_max[i] + 1] = (
                scores[i]
            )

        result = result.unsqueeze(0)
    return result


def pixel_wise_target(data_batch_prepro, num_classes: int = 80):
    labels = data_batch_prepro["data_samples"][0].gt_instances.labels
    bboxes = data_batch_prepro["data_samples"][0].gt_instances.bboxes
    _, img_h, img_w = data_batch_prepro.get("inputs")[0].shape
    result = torch.zeros((num_classes, img_h, img_w), device="cuda")

    # Clip bboxes to image dimensions
    x_min = torch.clamp(bboxes[:, 0], 0, img_w - 1).long()
    y_min = torch.clamp(bboxes[:, 1], 0, img_h - 1).long()
    x_max = torch.clamp(bboxes[:, 2], 0, img_w - 1).long()
    y_max = torch.clamp(bboxes[:, 3], 0, img_h - 1).long()

    # Vectorized bbox filling
    for i in range(bboxes.size(0)):
        result[labels[i], y_min[i] : y_max[i] + 1, x_min[i] : x_max[i] + 1] = 1

    return result.unsqueeze(0)


def preprocess_data(data_batch: dict, runner):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)
    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    mean = runner.model.data_preprocessor.mean
    std = runner.model.data_preprocessor.std
    images = denorm(images, mean, std)
    return data_batch_prepro, images


def cospgd_attack(
    data_batch: dict,
    runner: Runner,
    steps: int,
    epsilon: float,
    alpha: float,
    targeted: bool,
    random_start: bool,
    evaluators: List[Evaluator],
    criterion: Callable[..., torch.Tensor] = F.cross_entropy,
):
    global image_counter
    global count_early_stopping
    image_counter += 1
    stopped = False

    data_batch_prepro, images = preprocess_data(data_batch, runner)
    adv_images = images.clone().float().detach().to("cuda")
    evaluator_process(evaluators[0], data_batch_prepro, runner)

    labels = pixel_wise_target(data_batch_prepro, num_classes=80)

    if targeted:
        labels = torch.ones_like(labels)

    # Assume norm == inf for now
    adv_images = cospgd.functions.init_linf(
        adv_images, epsilon=epsilon, clamp_min=0, clamp_max=255
    )
    adv_images.requires_grad = True
    data_batch_prepro["inputs"][0] = adv_images
    runner.model.requires_grad_(True)

    for step in range(steps):
        with torch.enable_grad():
            preds = pixel_wise_pred(
                data_batch_prepro, runner.model, adv_images
            ).requires_grad_(True)

        if preds is not None:
            adv_images.requires_grad = True
            data_batch_prepro["inputs"][0] = adv_images  # unnormalized
            runner.model.zero_grad()
            adv_images.grad = None
            with torch.enable_grad():
                loss = criterion(preds, labels, reduction="none")
                loss = cospgd_scale(
                    predictions=preds,
                    labels=labels,
                    loss=loss,
                    num_classes=80,
                    targeted=targeted,
                )
                loss = loss.mean()
            loss.backward(inputs=adv_images)

            if adv_images.grad is None and not stopped:
                stopped = True
                count_early_stopping += 1
                print(
                    f"No gradient: stopping early. Skipped {count_early_stopping} images of {image_counter} at attack step {step}"
                )
            else:
                # Assume norm == "inf" for now
                perturbed_image = cospgd.functions.step_inf(
                    perturbed_image=adv_images,
                    epsilon=epsilon,
                    data_grad=adv_images.grad,
                    orig_image=images,
                    alpha=alpha,
                    targeted=targeted,
                    clamp_min=0,
                    clamp_max=255,
                )

                data_batch_prepro["inputs"][0] = perturbed_image
                adv_images = perturbed_image.detach()

                transforms.Normalize(
                    runner.model.data_preprocessor.mean,
                    runner.model.data_preprocessor.std,
                    inplace=True,
                )(data_batch_prepro["inputs"][0])
        elif not stopped:
            stopped = True
            count_early_stopping += 1
            print(
                f"No predictions: stopping early. Skipped {count_early_stopping} images of {image_counter} at attack step {step}"
            )
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
    data_batch_prepro, images = preprocess_data(data_batch, runner)
    adv_images = images.clone().float().detach().to("cuda")
    evaluator_process(evaluators[0], data_batch_prepro, runner)

    if targeted:
        raise NotImplementedError

    if random_start:
        raise NotImplementedError

    for step in range(steps):
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        runner.model.training = True  # avoid missing arguments error in some models
        losses = runner.model(**data_batch_prepro, mode="loss")
        loss, _ = runner.model.parse_losses(losses)

        runner.model.zero_grad()
        adv_images.grad = None

        loss.backward(retain_graph=True)
        grad = adv_images.grad
        assert grad is not None

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

        data_batch_prepro["inputs"][0] = adv_images
        transforms.Normalize(
            runner.model.data_preprocessor.mean,
            runner.model.data_preprocessor.std,
            inplace=True,
        )(data_batch_prepro["inputs"][0])
        runner.model.training = False  # avoid missing arguments error in some models
        evaluator_process(evaluators[step + 1], data_batch_prepro, runner)

    return data_batch_prepro


def evaluator_process(evaluator, data_batch_prepro, runner):
    runner.model.training = False  # avoid missing arguments error in some models

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
    data_batch_prepro, images = preprocess_data(data_batch, runner)
    adv_images = images.clone().float().detach().to("cuda")

    evaluator_process(evaluators[0], data_batch_prepro, runner)

    # Get gradient
    adv_images.requires_grad = True
    data_batch_prepro["inputs"][0] = adv_images

    runner.model.training = True  # avoid missing arguments error in some models
    losses = runner.model(**data_batch_prepro, mode="loss")

    loss, _ = runner.model.parse_losses(losses)
    runner.model.zero_grad()
    loss.backward(retain_graph=True)
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
    else:
        raise NotImplementedError

    # Clipping to maintain [0, 255] range
    adv_images = torch.clamp(images + delta, 0, 255).detach()

    # Return the perturbed image
    adv_images.requires_grad = False
    data_batch_prepro["inputs"][0] = adv_images
    transforms.Normalize(
        runner.model.data_preprocessor.mean,
        runner.model.data_preprocessor.std,
        inplace=True,
    )(data_batch_prepro["inputs"][0])
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
    data_batch_prepro, images = preprocess_data(data_batch, runner)
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
        else:
            raise NotImplementedError
        adv_images = torch.clamp(images + delta, 0, 255).detach()

        data_batch_prepro["inputs"][0] = adv_images
        transforms.Normalize(
            runner.model.data_preprocessor.mean,
            runner.model.data_preprocessor.std,
            inplace=True,
        )(data_batch_prepro["inputs"][0])
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
    batch_size: int = 1,
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
                    "model_name": model_name,
                    "batch_size": batch_size,
                },
                "group": model_name,
                "tags": ["batch size test"],
            },
        )
    )
    cfg.val_dataloader.batch_size = batch_size

    if DEBUG:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print(f"torch cuda version: {torch.version.cuda}")
        # cfg.val_dataloader.dataset.indices = 500  # reduce dataset size for debugging

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
        default="cospgd",
        choices=["pgd", "fgsm", "bim", "cospgd", "none"],
        help="Type of attack (default: cospgd)",
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
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of images per batch"
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
        batch_size=args.batch_size,
    )
