"""Validate optical flow estimation performance on standard datasets."""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pdb

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Normalize
import ptlflow
from ptlflow_attacked.ptlflow import get_model, get_model_reference
from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from ptlflow_attacked.ptlflow.utils import flow_utils
from ptlflow_attacked.ptlflow.utils.io_adapter import IOAdapter
from ptlflow_attacked.ptlflow.utils.utils import (
    add_datasets_to_parser,
    config_logging,
    get_list_of_available_models_list,
    tensor_dict_to_numpy,
)
from attacks.attack_utils.utils import get_image_tensors, get_image_grads, replace_images_dic, get_flow_tensors
from attacks.fgsm import fgsm
from attacks.apgd import apgd
from attacks.bim_pgd_cospgd import bim_pgd_cospgd
from attacks.fab import fab
from ptlflow_attacked.validate import validate_one_dataloader, generate_outputs, _get_model_names
# Import cosPGD functions
import torch.nn as nn
# Attack parameters
epsilon = 8 / 255
norm = "inf"
alpha = 0.01
iterations = 3
# criterion = nn.CrossEntropyLoss(reduction="none")
# criterion = nn.MSELoss(reduction="none")
loss_function = "epe"
targeted = False
batch_size = 1


# PCFA parameters
import torch.optim as optim
delta_bound=0.005


config_logging()


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=["all", "select"] + get_list_of_available_models_list(),
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="none",
        choices=["fgsm", "bim", "pgd", "cospgd", "ffgsm", "apgd", "fab", "none"],
        help="Name of the attack to use.",
    )
    parser.add_argument(
        "--attack_norm",
        type=str,
        default=norm,
        choices=["two", "inf"],
        help="Set norm to use for adversarial attack.",
    )
    parser.add_argument(
        "--attack_epsilon",
        type=float,
        default=epsilon,
        help="Set epsilon to use for adversarial attack.",
    )
    parser.add_argument(
        "--attack_delta_bound",
        type=float,
        default=delta_bound,
        help="Set delta bound to use for PCFA.",
    )
    parser.add_argument('--boxconstraint', default='change_of_variables', choices=['clipping', 'change_of_variables'],
                help="the way to enfoce the box constraint on the distortion. Options: 'clipping', 'change_of_variables'.")
    parser.add_argument(
        "--attack_iterations",
        type=int,
        default=iterations,
        help="Set number of iterations for adversarial attack.",
    )
    parser.add_argument(
        "--attack_alpha",
        type=float,
        default=alpha,
        help="Set epsilon to use for adversarial attack.",
    )
    parser.add_argument(
    "--attack_targeted",
    type=bool,
    default=targeted,
    help="Set if adversarial attack should be targeted.",
    )
    parser.add_argument(
    "--attack_loss",
    type=str,
    default=loss_function,
    help="Set the name of the used loss function (mse, epe)",
    )
    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with model=select. The select mode can be used to run the validation on multiple models "
            "at once. Put a list of model names here separated by spaces."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with model=all. A list of model names that will not be validated."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/validate")),
        help="Path to the directory where the validation results will be saved.",
    )
    parser.add_argument(
        "--write_outputs",
        action="store_true",
        help="If set, the estimated flow is saved to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, the results are shown on the screen.",
    )
    parser.add_argument(
        "--flow_format",
        type=str,
        default="original",
        choices=["flo", "png", "original"],
        help=(
            "The format to use when saving the estimated optical flow. If 'original', then the format will be the same "
            + "one the dataset uses for the groundtruth."
        ),
    )
    parser.add_argument(
        "--max_forward_side",
        type=int,
        default=None,
        help=(
            "If max(height, width) of the input image is larger than this value, then the image is downscaled "
            "before the forward and the outputs are bilinearly upscaled to the original resolution."
        ),
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help=("Multiply the input image by this scale factor before forwarding."),
    )
    parser.add_argument(
        "--max_show_side",
        type=int,
        default=1000,
        help=(
            "If max(height, width) of the output image is larger than this value, then the image is downscaled "
            "before showing it on the screen."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples per dataset will be used for calculating the metrics."
        ),
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="To be combined with model all or select. Iterates over the list of models in reversed order",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="If set, stores the previous estimation to be used a starting point for prediction.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    parser.add_argument(
        "--seq_val_mode",
        type=str,
        default="all",
        choices=("all", "first", "middle", "last"),
        help=(
            "Used only when the model predicts outputs for more than one frame. Select which predictions will be used for evaluation."
        ),
    )
    parser.add_argument(
        "--write_individual_metrics",
        action="store_true",
        help="If set, save a table of metrics for every image.",
    )
    return parser


def attack(args: Namespace, model: BaseModel) -> pd.DataFrame:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the metric results.

    See Also
    --------
    ptlflow.models.base_model.base_model.BaseModel : The parent class of the available models.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if args.fp16:
            model = model.half()

    dataloaders = model.val_dataloader()
    dataloaders = {
        model.val_dataloader_names[i]: dataloaders[i] for i in range(len(dataloaders))
    }

    metrics_df = pd.DataFrame()
    metrics_df["model"] = [args.model]
    metrics_df["checkpoint"] = [args.pretrained_ckpt]

    for dataset_name, dl in dataloaders.items():
        if args.attack == "none":
            metrics_mean = validate_one_dataloader(args, model, dl, dataset_name)
        else: 
            metrics_mean = attack_one_dataloader(args, model, dl, dataset_name)
        metrics_df[[f"{dataset_name}-{k}" for k in metrics_mean.keys()]] = list(
            metrics_mean.values()
        )
        args.output_path.mkdir(parents=True, exist_ok=True)
        metrics_df.T.to_csv(args.output_path / f"metrics_{args.val_dataset}_{args.attack}_{args.attack_targeted}.csv", header=False)
    metrics_df = metrics_df.round(3)
    return metrics_df


def attack_list_of_models(args: Namespace) -> None:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the list of models and the validation.
    """
    metrics_df = pd.DataFrame()

    model_names = _get_model_names(args)
    if args.reversed:
        model_names = reversed(model_names)

    exclude = args.exclude
    if exclude is None:
        exclude = []
    for mname in model_names:
        if mname in exclude:
            continue

        logging.info(mname)
        model_ref = ptlflow.get_model_reference(mname)

        if hasattr(model_ref, "pretrained_checkpoints"):
            ckpt_names = model_ref.pretrained_checkpoints.keys()
            for cname in ckpt_names:
                try:
                    logging.info(cname)
                    parser_tmp = model_ref.add_model_specific_args(parser)
                    args = parser_tmp.parse_args()

                    args.model = mname
                    args.pretrained_ckpt = cname

                    model_id = args.model
                    if args.pretrained_ckpt is not None:
                        model_id += f"_{args.pretrained_ckpt}"
                    args.output_path = Path(args.output_path) / model_id

                    model = get_model(mname, cname, args)
                    instance_metrics_df = attack(args, model)
                    metrics_df = pd.concat([metrics_df, instance_metrics_df])
                    args.output_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.reversed:
                        metrics_df.to_csv(
                            args.output_path.parent / "metrics_all_rev.csv", index=False
                        )
                    else:
                        metrics_df.to_csv(
                            args.output_path.parent / "metrics_all.csv", index=False
                        )
                except Exception as e:  # noqa: B902
                    logging.warning("Skipping model %s due to exception %s", mname, e)
                    break


@torch.enable_grad()
def attack_one_dataloader(
    args: Namespace,
    model: BaseModel,
    dataloader: DataLoader,
    dataloader_name: str,
) -> Dict[str, float]:
    """Perform adversarial attack for all examples of one dataloader.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.
    dataloader : DataLoader
        The dataloader for the validation.
    dataloader_name : str
        A string to identify this dataloader.

    Returns
    -------
    Dict[str, float]
        The average metric values for this dataloader.
    """
    metrics_sum = {}

    metrics_individual = None
    if args.write_individual_metrics:
        metrics_individual = {"filename": [], "epe": [], "outlier": []}

    losses = torch.zeros(len(dataloader))
    with tqdm(dataloader) as tdl:
        prev_preds = None
        for i, inputs in enumerate(tdl):
            if args.scale_factor is not None:
                scale_factor = args.scale_factor
            else:
                scale_factor = (
                    None
                    if args.max_forward_side is None
                    else float(args.max_forward_side) / min(inputs["images"].shape[-2:])
                )

            io_adapter = IOAdapter(
                model,
                inputs["images"].shape[-2:],
                target_scale_factor=scale_factor,
                cuda=torch.cuda.is_available(),
                fp16=args.fp16,
            )
            inputs = io_adapter.prepare_inputs(inputs=inputs, image_only=True)
            inputs["prev_preds"] = prev_preds

            if inputs["images"].max() > 1.0:
                args.attack_epsilon = args.attack_epsilon*255

            targeted_inputs = None
            if args.attack_targeted:
                targeted_flow_tensor = torch.zeros_like(inputs["flows"])
                targeted_inputs = inputs.copy()
                targeted_inputs["flows"] = targeted_flow_tensor
                with torch.no_grad():
                    orig_preds = model(inputs)

            # TODO: figure out what to do with scaled images and labels
            match args.attack: # Commit adversarial attack
                case "fgsm":
                    # inputs["images"] = fgsm(args, inputs, model)
                    images, labels, preds, placeholder = fgsm(args, inputs, model, targeted_inputs)
                case "pgd":
                    # inputs["images"] = cos_pgd(args, inputs, model)
                    images, labels, preds, losses[i] = bim_pgd_cospgd(args, inputs, model, targeted_inputs)
                case "cospgd":
                    # inputs["images"] = cos_pgd(args, inputs, model)
                    images, labels, preds, losses[i] = bim_pgd_cospgd(args, inputs, model, targeted_inputs)
                case "bim":
                    # inputs["images"] = cos_pgd(args, inputs, model)
                    images, labels, preds, losses[i] = bim_pgd_cospgd(args, inputs, model, targeted_inputs)
                case "apgd":
                    # inputs["images"] = fgsm(args, inputs, model)
                    images, labels, preds, placeholder = apgd(args, inputs, model, targeted_inputs)
                case "fab":
                    images, labels, preds, placeholder = fab(args, inputs, model, targeted_inputs)
                case "none":
                    preds = model(inputs)
            
            if args.warm_start:
                if (
                    "is_seq_start" in inputs["meta"]
                    and inputs["meta"]["is_seq_start"][0]
                ):
                    prev_preds = None
                else:
                    prev_preds = preds
                    for k, v in prev_preds.items():
                        if isinstance(v, torch.Tensor):
                            prev_preds[k] = v.detach()

            inputs = io_adapter.unscale(inputs, image_only=True)
            preds = io_adapter.unscale(preds)

            if inputs["flows"].shape[1] > 1 and args.seq_val_mode != "all":
                if args.seq_val_mode == "first":
                    k = 0
                elif args.seq_val_mode == "middle":
                    k = inputs["images"].shape[1] // 2
                elif args.seq_val_mode == "last":
                    k = inputs["flows"].shape[1] - 1
                for key, val in inputs.items():
                    if key == "meta":
                        inputs["meta"]["image_paths"] = inputs["meta"]["image_paths"][
                            k : k + 1
                        ]
                    elif key == "images":
                        inputs[key] = val[:, k : k + 2]
                    elif isinstance(val, torch.Tensor) and len(val.shape) == 5:
                        inputs[key] = val[:, k : k + 1]

            # metrics = model.val_metrics(preds, inputs)
            if args.attack_targeted:
                metrics = model.val_metrics(preds, targeted_inputs)
                metrics_ground_truth = model.val_metrics(preds, inputs)
                metrics_orig_preds = model.val_metrics(preds, orig_preds)
                metrics['val/epe_ground_truth'] = metrics_ground_truth['val/epe']
                metrics['val/epe_orig_preds'] = metrics_orig_preds['val/epe']
            else:
                metrics = model.val_metrics(preds, inputs)

            for k in metrics.keys():
                if metrics_sum.get(k) is None:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += metrics[k].item()
            tdl.set_postfix(
                epe=metrics_sum["val/epe"] / (i + 1),
                outlier=metrics_sum["val/outlier"] / (i + 1),
            )

            filename = ""
            if "sintel" in inputs["meta"]["dataset_name"][0].lower():
                filename = f'{Path(inputs["meta"]["image_paths"][0][0]).parent.name}/'
            filename += Path(inputs["meta"]["image_paths"][0][0]).stem

            if metrics_individual is not None:
                metrics_individual["filename"].append(filename)
                metrics_individual["epe"].append(metrics["val/epe"].item())
                metrics_individual["outlier"].append(metrics["val/outlier"].item())

            if args.attack_targeted:
                generate_outputs(
                args, targeted_inputs, preds, dataloader_name, i, targeted_inputs.get("meta")
            )
            else:
                generate_outputs(
                    args, inputs, preds, dataloader_name, i, inputs.get("meta")
                )

            if args.max_samples is not None and i >= (args.max_samples - 1):
                break

    if args.write_individual_metrics:
        ind_df = pd.DataFrame(metrics_individual)
        args.output_path.mkdir(parents=True, exist_ok=True)
        ind_df.to_csv(
            Path(args.output_path) / f"{dataloader_name}_epe_outlier.csv", index=None
        )

    metrics_mean = {}
    for k, v in metrics_sum.items():
        metrics_mean[k] = v / len(dataloader)
    return metrics_mean


def attack_pcfa(args: Namespace, inputs: Dict[str, torch.Tensor], model: BaseModel, targeted_inputs: Optional[Dict[str, torch.Tensor]]):
    """
    Performs an PCFA attack on a given model and for all images of a specified dataset.
    """

    optim_mu = 2500./args.delta_bound
        
    optimizer_lr = args.delta_bound

    eps_box = 1e-7

    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Make sure the model is not trained:
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # If dataset has ground truth
    has_gt = True

    # if args.attack_targeted:
    #     labels = targeted_inputs["flows"].squeeze(0)
    # else:
    flow = inputs["flows"].squeeze(0)

    orig_image_1 = inputs["images"][0].clone()[0].unsqueeze(0)
    orig_image_2 = inputs["images"][0].clone()[1].unsqueeze(0)

    image_1, image_2 = get_image_tensors(inputs)

    aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt, aee_adv_tgt, aee_adv_pred, l2_delta1, l2_delta2, l2_delta12, aee_adv_tgt_min_val, aee_adv_pred_min_val, delta12_min_val = pcfa_attack(model, image_1, image_2, flow, eps_box, device, optimizer_lr, has_gt, optim_mu, args)
    

def pcfa_attack(model, image1, image2, flow, eps_box, device, optimizer_lr, has_gt, optim_mu, args):
    """Subroutine to optimize a PCFA perturbation on a given image pair. For a specified number of steps.

    Args:
        model (torch.nn.module):
            a pytorch model which is set to eval and which is implemented in ownutilities.['preprocess_img','compute_flow','postprocess_flow']
        image1 (torch.tensor):
            image 1 of a scene with dimensions (b,3,h,w)
        image2 (torch.tensor):
            image 2 of a scene with dimensions (b,3,h,w)
        flow (torch.tensor):
            intial (unattacked) flow field (resulting from img1 and img2) which can be used to log the current effect induced by the patch (same spatial dimension as images!)
        batch (int):
            current image counter in the enumeration of the test set
        distortion_folder (string):
            name for a folder that will be created to hold data and visualizations of the distortions that are trained during the PCFA
        eps_box (float):
            relaxation of the box constraint due to numerical reasons (try 1e-07)
        device (torch.device):
            Select the device for the images.
        optimizer_lr (float):
            optimizer learning rate (try 5e-03)
        has_gt (boolean):
            is the ground truth known
        optim_mu (float):
            regularization parameter of the constraint in the unconstraing optimization (try 5e05)
        args (Namespace):
            command line arguments


    Returns:
        aee_gt (float):
            Average Endpoint Error of the ground truth w.r.t. zero flow (none if has_gt is false)
        aee_tgt (float):
            Average Endpoint Error of the target flow w.r.t. zero flow
        aee_gt_tgt (float):
            Average Endpoint Error of the ground truth towards the target flow (none if has_gt is false)
        aee_adv_tgt (float):
            Average Endpoint Error of the adversarial predicted flow (after the perturbation is added) towards the target flow
        aee_adv_pred (float):
            Average Endpoint Error of the adversarial predicted flow (after the perturbation is added) towards the initially predicted flow
        l2_delta2 (float):
            l2 error of the preturbation on image1
        l2_delta2 (float):
            l2 error of the preturbation on image2
        l2_delta12 (float):
            scalar average L2-norm of two perturbations
        aee_adv_tgt_min_val (float):
            minimal (best attack) Average Endpoint Error of the adversarial predicted flow towards the target flow, while also not violating the constraint, i.e. delta12 < args.delta_bound
        aee_adv_pred_min_val (float):
            best attack Average Endpoint Error of the adversarial predicted flow towards the initially predicted flow, while also not violating the constraint
        delta12_min_val (float):
            scalar average L2-norm of two perturbations of the best attack

    Extended Output:
        Files in .npy and .png format of initial images and flow, perturbations, and the adversarial images and flow are saved.
        The best attack perturbation images and flows are labeled with "*best*" and correspond to the values of "*_min*".
    """
    torch.autograd.set_detect_anomaly(True)

    image1, image2 = image1.to(device), image2.to(device)
    flow = flow.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    images_max = torch.max(image1, image2).detach().to(device)
    images_min = torch.min(image1, image2).detach().to(device)

    # initialize perturbation and auxiliary variables:
    delta1 = torch.zeros_like(image1)
    delta2 = torch.zeros_like(image2)
    delta1 = delta1.to(device)
    delta2 = delta2.to(device)

    nw_input1 = None
    nw_input2 = None
    nw_delta = None

    flow_pred_init = None

    # Set up the optimizer and variables if individual perturbations delta1 and delta2 for images 1 and 2 should be trained
    delta1.requires_grad = False
    delta2.requires_grad = False

    if args.boxconstraint in ['change_of_variables']:
        nw_input1 = torch.atanh( 2. * (1.- eps_box) * (image1 + delta1) - (1 - eps_box)  )
        nw_input2 = torch.atanh( 2. * (1.- eps_box) * (image2 + delta2) - (1 - eps_box)  )
    else:
        nw_input1 = image1 + delta1
        nw_input2 = image2 + delta2

    nw_input1.requires_grad = True
    nw_input2.requires_grad = True

    optimizer = optim.LBFGS([nw_input1, nw_input2], max_iter=10)

    #TODO: ab hier weitermachen
    # Predict the flow
    flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
    [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
    flow_pred = flow_pred.to(device)

    # define the initial flow, the target, and update mu
    flow_pred_init = flow_pred.detach().clone()
    flow_pred_init.requires_grad = False


    # define target (potentially based on first flow prediction)
    # define attack target
    target = targets.get_target(args.target, flow_pred_init, flow_target_scale=args.flow_target_scale, custom_target_path=args.custom_target_path, device=device)
    target = target.to(device)
    target.requires_grad = False

    # ToDo: Write test that checks for first image of sintel subsplit and RAFT, if aee_gt (0.225551) and aee_tgt (0.92254) are correct.
    # Some EPE statistics for the unattacked flow
    aee_tgt            = logging.calc_metrics_const(target, flow_pred_init)
    aee_gt_tgt, aee_gt = logging.calc_metrics_const_gt(target, flow_pred_init, flow) if has_gt else (None, None)

    logging.log_metrics(curr_step, ("aee_pred-tgt", aee_tgt),
                                   ("aee_gt-tgt", aee_gt_tgt),
                                   ("aee_pred-gt", aee_gt))

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    delta_below_threshold=False
    delta12_min_val = float('inf')
    aee_adv_tgt_min_val = float('inf')
    aee_adv_pred_min_val = 0.
    l2_delta1, l2_delta2, l2_delta12 = 0, 0, 0
    delta1_min = torch.ones_like(image1).detach()
    delta2_min = torch.ones_like(image2).detach()
    flow_pred_min = flow_pred_init.detach().clone()

    for steps in range(args.steps):

        curr_step = batch*args.steps + steps
        log_metric(key="batch", value=batch, step=curr_step)
        log_metric(key="steps", value=steps, step=curr_step)
        log_metric(key="epoch", value=0, step=curr_step)
        #print("step: " + str(steps))

        # Calculate the deltas from the quantities that go into the network
        if args.joint_perturbation:
            delta1, delta2 = extract_deltas_joint(nw_delta, images_max, images_min)
        else:
            delta1, delta2 = extract_deltas(nw_input1, nw_input2, image1, image2, args.boxconstraint, eps_box=eps_box)

        # Calculate the loss
        if args.delta_bound > 0.0:
            # print('using loss delta_bound')
            loss = losses.loss_delta_constraint(flow_pred, target, delta1, delta2, device, delta_bound=args.delta_bound, mu=optim_mu,  f_type=args.loss)

            similarity_term = ownutilities.torchfloat_to_float64(losses.get_loss(args.loss, flow_pred, target))
            penalty_term = ownutilities.torchfloat_to_float64(losses.relu_penalty(delta1, delta2, device, args.delta_bound))
            log_metric(key="term_similarity", value=similarity_term, step=curr_step)
            log_metric(key="term_penalty", value=penalty_term, step=curr_step)
            log_metric(key="term_penalty_mu", value=penalty_term*optim_mu, step=curr_step)
        else:
            # print('using loss weighted')
            loss = losses.loss_weighted(flow_pred, target, delta1, delta2, c=args.weighting, f_type=args.loss)


        # Update the optimization parameters
        loss.backward()

        if args.optimizer in ['Adam', 'SGD']:
            optimizer.step()

        elif args.optimizer in ['LBFGS']:

            def closure():
                optimizer.zero_grad()
                if args.joint_perturbation:
                    flow_closure = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True, delta1=nw_delta)
                else:
                    flow_closure = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
                [flow_closure] = ownutilities.postprocess_flow(args.net, padder, flow_closure)
                flow_closure = flow_closure.to(device)
                if args.joint_perturbation:
                    delta1_closure, delta2_closure = extract_deltas_joint(nw_delta, images_max, images_min)
                else:
                    delta1_closure, delta2_closure = extract_deltas(nw_input1, nw_input2, image1, image2, args.boxconstraint, eps_box=eps_box)
                loss_closure = losses.loss_delta_constraint(flow_closure, target, delta1_closure, delta2_closure, device, delta_bound=args.delta_bound, mu=optim_mu,  f_type=args.loss)
                loss_closure.backward()
                return loss_closure

            # Update the optimization parameters
            optimizer.step(closure)
        else:
            raise RuntimeWarning('Unknown optimizer, no optimization step was performed')

        # calculate the magnitude of the updated distortion, and with it the new network inputs:
        if args.joint_perturbation:
            delta1, delta2 = extract_deltas_joint(nw_delta, images_max, images_min)
            if args.boxconstraint in ['change_of_variables']:
                raise ValueError("Training a --joint_perturbation with --boxconstraint=change_of_variables is not defined. Please use --boxconstraint=clipping.")
            else:
                nw_input1 = image1
                nw_input2 = image2
        else:
            delta1, delta2 = extract_deltas(nw_input1, nw_input2, image1, image2, args.boxconstraint, eps_box=eps_box)
            # The nw_inputs remain unchanged in this case, and can be directly fed into the network again for further perturbation training

        # Re-predict flow with the perturbed image, and update the flow prediction for the next iteration
        if args.joint_perturbation:
            flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True, delta1=nw_delta)
        else:
            flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
        [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
        flow_pred = flow_pred.to(device)

        # More AEE statistics, now for attacked images
        aee_adv_tgt, aee_adv_pred = logging.calc_metrics_adv(flow_pred, target, flow_pred_init)
        aee_adv_gt                = logging.calc_metrics_adv_gt(flow_pred, flow) if has_gt else None
        logging.log_metrics(curr_step, ("aee_predadv-tgt", aee_adv_tgt),
                                       ("aee_pred-predadv", aee_adv_pred),
                                       ("aee_predadv-gt", aee_adv_gt))

        l2_delta1, l2_delta2, l2_delta12 = logging.calc_delta_metrics(delta1, delta2, curr_step)
        logging.log_metrics(curr_step, ("l2_delta1", l2_delta1),
                                       ("l2_delta2", l2_delta2),
                                       ("l2_delta-avg", l2_delta12))

        update_minima = False
        if not delta_below_threshold:
            if l2_delta12 < delta12_min_val or (l2_delta12 == delta12_min_val and aee_adv_tgt < aee_adv_tgt_min_val):

                update_minima = True
                if l2_delta12 <= args.delta_bound:
                    delta_below_threshold = True
        else:
            if l2_delta12 <= args.delta_bound and aee_adv_tgt < aee_adv_tgt_min_val:
                update_minima = True

        if update_minima:
            delta12_min_val = l2_delta12
            aee_adv_tgt_min_val = aee_adv_tgt
            aee_adv_pred_min_val = aee_adv_pred
            delta1_min = delta1.detach().clone()
            delta2_min = delta2.detach().clone()
            flow_pred_min = flow_pred.detach().clone()

        logging.log_metrics(curr_step, ("aee_pred-tgt_min", aee_adv_tgt_min_val),
                                   ("l2_delta-avg_min", delta12_min_val),
                                   ("aee_pred-predadv_min", aee_adv_pred_min_val))


    # Final saving of images:
    # ToDo: How can the joint_perturbation flag here be handeled? Even if in theory delta1==delta2, due to clipping the distortions can be different for both images...
    # No, because this case was explicitely treated by the cropped common perturbation, which fits both images now
    if ((batch % args.save_frequency == 0 and not args.small_save) or (args.small_save and batch < 32)) and not args.no_save:

        logging.save_tensor(delta1, "delta1_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(delta2, "delta2_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(delta1_min, "delta1_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(delta2_min, "delta2_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(image1, "image1", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(image2, "image2", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(target, "target", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_pred, "flow_pred_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_pred_min, "flow_pred_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_pred_init, "flow_pred_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        if has_gt:
            logging.save_tensor(flow, "flow_gt", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)


        logging.save_image(image1, batch, distortion_folder, image_name='image1', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image2, batch, distortion_folder, image_name='image2', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image1+delta1_min, batch, distortion_folder, image_name='image1_delta_best', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image2+delta2_min, batch, distortion_folder, image_name='image2_delta_best', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)


        max_delta = np.max([ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta1_min))),
                            ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta2_min)))])

        logging.save_image(delta1_min, batch, distortion_folder, image_name='delta1_best', unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
        if not args.joint_perturbation:
            logging.save_image(delta2_min, batch, distortion_folder, image_name='delta2_best', unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)


        max_flow_gt = 0
        if has_gt:
            max_flow_gt = ownutilities.maximum_flow(flow)
        max_flow = np.max([max_flow_gt,
                           ownutilities.maximum_flow(flow_pred_init),
                           ownutilities.maximum_flow(flow_pred_min)])

        logging.save_flow(flow_pred_min, batch, distortion_folder, flow_name='flow_pred_best', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(flow_pred_init, batch, distortion_folder, flow_name='flow_pred_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(target, batch, distortion_folder, flow_name='flow_target', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        if has_gt:
            logging.save_flow(flow, batch, distortion_folder, flow_name='flow_gt', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

    return aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt, aee_adv_tgt, aee_adv_pred, l2_delta1, l2_delta2, l2_delta12, aee_adv_tgt_min_val, aee_adv_pred_min_val, delta12_min_val



# From FlowUnderAttack
def epe(flow1, flow2):
    """"
    Compute the  endpoint errors (EPEs) between two flow fields.
    The epe measures the euclidean- / 2-norm of the difference of two optical flow vectors
    (u0, v0) and (u1, v1) and is defined as sqrt((u0 - u1)^2 + (v0 - v1)^2).

    Args:
        flow1 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component
        flow2 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component

    Raises:
        ValueError: dimensons not valid

    Returns:
        float: scalar average endpoint error
    """
    diff_squared = (flow1 - flow2)**2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=0).sqrt()
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=1).sqrt()
    else:
        raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(flow1.size()) + " and " + str(flow1.size()))
    return epe


if __name__ == "__main__":
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] not in ["-h", "--help", "all", "select"]:
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    add_datasets_to_parser(parser, "./ptlflow_attacked/datasets.yml")

    args = parser.parse_args()

    if args.model not in ["all", "select"]:
        model_id = args.model
        if args.pretrained_ckpt is not None:
            model_id += f"_{Path(args.pretrained_ckpt).stem}"
        if args.max_forward_side is not None:
            model_id += f"_maxside{args.max_forward_side}"
        if args.scale_factor is not None:
            model_id += f"_scale{args.scale_factor}"
        args.output_path = Path(args.output_path) / model_id
        model = get_model(sys.argv[1], args.pretrained_ckpt, args)
        args.output_path.mkdir(parents=True, exist_ok=True)

        attack(args, model)
    else:
        attack_list_of_models(args)


