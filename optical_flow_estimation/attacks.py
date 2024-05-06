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
from datetime import datetime

import os
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

from attacks.fgsm import fgsm
from attacks.apgd import apgd
from attacks.bim_pgd_cospgd import bim_pgd_cospgd
from attacks.fab import fab
from attacks.pcfa import pcfa
import ast
from attacks.weather import weather_ds
from attacks.common_corruptions import common_corrupt
from attacks.attack_utils.attack_args_parser import AttackArgumentParser
from attacks.attack_utils.attack_args_parser import (
    attack_targeted_string,
    attack_arg_string,
)
from ptlflow_attacked.validate import (
    validate_one_dataloader,
    generate_outputs,
    _get_model_names,
)
from torch.nn.functional import cosine_similarity
from attacks.attack_utils.utils import get_flow_tensors
# Import cosPGD functions
import torch.nn as nn


# Attack parameters
epsilon = 8 / 255
norm = "inf"
alpha = 0.01
iterations = 3
loss_function = "epe"
targeted = False
batch_size = 1


weather_steps=750
lr=0.001 
alph_motion=1000
alph_motionoffset=1000


learn_offset=True 
learn_motionoffset=True
learn_color=True  
learn_transparency=True     

# PCFA parameters
import torch.optim as optim

delta_bound = 0.005


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
        nargs="*",
        choices=[
            "fgsm",
            "bim",
            "pgd",
            "cospgd",
            "ffgsm",
            "apgd",
            "fab",
            "pcfa",
            "weather",
            "common_corruptions",
            "none",
        ],
        help="Name of the attack to use.",
    )
     # parser for weather
    parser.add_argument(
        "--weatherdat", 
        type=str,
        default=str(Path("/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/DistractingDownpour/particles_3000_npz")),
        help="the npz files."
        ) 
    parser.add_argument(
        "--weather_steps", 
        type=int,
        default=750,
        nargs="*",
        help="the number of optimization steps per image."
        )
    # parser.add_argument(
    #     "--lr", 
    #     type=float, 
    #     default=0.00001,
    #     nargs="*",
    #     help="learning rate for updating the distortion via stochastic gradient descent or Adam. Default: 0.001."
    #     )

    # parser.add_argument(
    #     "--loss",
    #     type=str,
    #     default="aee",
    #     nargs="*",
    #     choices=["aee", "mse", "cosim"],
    #     help="specify the loss function as one of 'aee', 'cosim' or 'mse'"
    #     )
    parser.add_argument(
        "--weather_learn_offset",
        default=True,
        type=ast.literal_eval,
        help="if specified, initial position of the particles will be optimized."
        )
    parser.add_argument(
        "--weather_learn_motionoffset",
        default=True,
        type=ast.literal_eval,
        help="if specified, the endpoint of the particle motion will be optimized (along with the starting point)."
        )
    parser.add_argument(
        "--weather_learn_color",
        default=True,
        type=ast.literal_eval,
        help="if specified, the color of the particle will be optimized."
        )
    parser.add_argument(
        "--weather_learn_transparency",
        default=True,
        type=ast.literal_eval,
        help="if specified, the transparency of the particles will be optimized.")
    parser.add_argument(
        "--weather_alph_motion",
        default=1000.,
        type=float,
        help="weighting for the motion loss.")  
    parser.add_argument(
        "--weather_alph_motionoffset",
        default=1000., 
        type=float,
        help="weighting for the motion offset loss."
        )     
    parser.add_argument(
        "--weather_data",
        default="/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation/DistractingDownpour/particles_3000_png",
        help="may specify a dataset that contains weather data (locations, masks, etc). It should have the same structure as the used dataset.")                             
    parser.add_argument(
        "--weather_dataset",
        default="Sintel",
        type= str,
        nargs="*",
        help="specify the dataset which should be used for evaluation"
        )
    parser.add_argument(
        "--weather_dataset_stage",
        default="evaluation",
        choices=["training", "evaluation"],
        help="specify the dataset stage ('training' or 'evaluation') that should be used."
        )
    parser.add_argument('--weather_small_run', action='store_true',
        help="for testing purposes: if specified the dataloader will on load 32 images")   
    parser.add_argument('--weather_optimizer', default="Adam",
                help="the optimizer used for the perturbations.")
    parser.add_argument('--weather_target', default='zero', choices=['zero'],
        help="specify the attack target.")
    # Sintel specific:
    parser.add_argument('--weather_dstype', default='final', choices=['clean', 'final'],
        help="[only sintel] specify the dataset type for the sintel dataset")
    parser.add_argument('--weather_single_scene', default='',
        help='if a scene name is added, only this sintel-scene is in the dataset.')
    parser.add_argument('--weather_from_scene', default='',
        help='if a scene is specified as from_scene, then all subsequent scenes (alphabetic order, including the specified scene) are added to the dataset.')
    parser.add_argument('--weather_frame_per_scene', default=0, type=int,
        help="the number of optimization scenes per sintel-sequence (if 0, all scenes per sequence are taken).")
    parser.add_argument('--weather_from_frame', default=0, type=int,
        help='For all scenes, the dataset starts from specified frame number. 0 takes all frames, 1 all except first, etc.')
    parser.add_argument('--weather_scene_scale', default=1.0, type=float,
        help="A global scaling to the scene depth. If the value is > 1, all scenes will appear bigger and more particles will show up in the foreground.")

    parser.add_argument('--weather_recolor', default=False, type=ast.literal_eval,
            help="If specified, all weather is recolored with the given r,g,b value (no variations).")
       # Motionblur arguments
    parser.add_argument('--weather_do_motionblur', default=True, type=ast.literal_eval,
            help="control if particles are rendered with motion blur (default=True).")
    parser.add_argument('--weather_motionblur_scale', default=.025, type=float,
            help="a scaling factor in [0,1], by which the motion blur is shortened. No motion blur appears for 0, while the full blur vector is used with 1. A full motion blur might need a higher number of motionblur_samples.")
    parser.add_argument('--weather_motionblur_samples', default=10, type=int,
            help="the number of flakes that is drawn per blurred flake. More samples are needed for faster objects or a larger motionblur_scale.")
        # Universal rendering arguments (also needed if flake data given)
    parser.add_argument('--weather_rendering_method', default='additive', choices=['meshkin', 'additive'],
            help="choose a method rendering the particle color. 'meshkin' use alpha-blending with order-independent transparency calculation, while 'additive' adds the color value to the image. Default: 'meshkin', choices: [meshkin, additive].")
    parser.add_argument('--weather_transparency_scale', default=1., type=float,
            help="a scaling factor, by which the tansparency for every particle is multiplied.")
        
    parser.add_argument("--weather_depth_check", default=False, type=ast.literal_eval,
        help="if specified, particles will not be rendered if behind an object.")
    parser.add_argument("--weather_depth_check_differentiable", type=ast.literal_eval, default=False, nargs="*",
        help="if specified, the rendering check for particle occlusion by objects is included into the compute graph.")
    parser.add_argument('--weather_unregistered_artifacts', default=True, type=ast.literal_eval,
        help="if True, artifacts are saved to the output folder but not registered. Saves time and memory during training.")
    parser.add_argument('--weather_output_folder', default='experiment_data',
        help="data that is logged during training and evaluation will be saved there")

    parser.add_argument(
        "--cc_name",
        type=str,
        default="gaussian_noise",
        nargs="*",
        choices=[
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
        ],
        help="Name of the common corruption to use on the input images.",
    )
    parser.add_argument(
        "--cc_severity",
        type=int,
        default=1,
        nargs="*",
        choices=[1,2,3,4,5],
        help="Severity of the common corruption to use on the input images.",
    )
    parser.add_argument(
        "--attack_norm",
        type=str,
        default=norm,
        nargs="*",
        choices=["two", "inf"],
        help="Set norm to use for adversarial attack.",
    )
    parser.add_argument(
        "--attack_epsilon",
        type=float,
        default=epsilon,
        nargs="*",
        help="Set epsilon to use for adversarial attack.",
    )
    parser.add_argument(
        "--pcfa_delta_bound",
        type=float,
        default=delta_bound,
        nargs="*",
        help="Set delta bound to use for PCFA.",
    )
    parser.add_argument(
        "--pcfa_boxconstraint",
        default="change_of_variables",
        nargs="*",
        choices=["clipping", "change_of_variables"],
        help="the way to enfoce the box constraint on the distortion. Options: 'clipping', 'change_of_variables'.",
    )
    parser.add_argument(
        "--pcfa_steps",
        default=5,
        type=int,
        nargs="*",
        help="the number of optimization steps per image (for non-universal perturbations only).",
    )
    parser.add_argument(
        "--apgd_rho",
        default=0.75,
        nargs="*",
        type=float,
        help="parameter for step-size update (Default: 0.75)",
    )
    parser.add_argument(
        "--apgd_n_restarts",
        default=1,
        nargs="*",
        type=int,
        help="number of random restarts. (Default: 1)",
    )
    parser.add_argument(
        "--apgd_eot_iter",
        default=1,
        nargs="*",
        type=int,
        help="number of iteration for EOT. (Default: 1)",
    )
    parser.add_argument(
        "--apgd_seed",
        default=0,
        nargs="*",
        type=int,
        help="random seed for the starting point. (Default: 0)",
    )
    parser.add_argument(
        "--apgd_steps",
        default=10,
        nargs="*",
        type=int,
        help="number of steps. (Default: 10)",
    )
    parser.add_argument(
        "--attack_iterations",
        type=int,
        default=iterations,
        nargs="*",
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
        type=attack_targeted_string,
        default=targeted,
        nargs="*",
        help="Set if adversarial attack should be targeted.",
    )
    parser.add_argument(
        "--attack_target",
        type=str,
        default="zero",
        nargs="*",
        choices=["zero", "negative"],
        help="Set the target for a tagreted attack.",
    )
    parser.add_argument(
        "--attack_loss",
        type=str,
        default=loss_function,
        nargs="*",
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
    parser.add_argument("--overwrite_output", type=bool, default=False)
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
    overwrite_flag = args.overwrite_output
    output_data = []
    output_data.append(
        (f"----ATTACK RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ----", "")
    )
    output_data.append(("model", args.model))
    output_data.append(("checkpoint", args.pretrained_ckpt))
    attack_args_parser = AttackArgumentParser(args)
    #print("args是：", args) #打印正常
    
    for attack_args in attack_args_parser:
        output_data.append(("attack_args", attack_arg_string(attack_args)))
        print(attack_args)
        for dataset_name, dl in dataloaders.items():
            if args.attack == "none":
                metrics_mean = validate_one_dataloader(args, model, dl, dataset_name)

            elif attack_args["attack"] == "weather":
                metrics_mean = attack_on_weather_dataloader(args,attack_args, model, dl, dataset_name)
            # else:
            #     metrics_mean = attack_one_dataloader(
            #         args, attack_args, model, dl, dataset_name
            #     )
            output_data.append(
                ("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            for k in metrics_mean.keys():
                output_data.append((f"{dataset_name}-{k}", metrics_mean[k]))
            metrics_df = pd.DataFrame(output_data, columns=["Type", "Value"])
            args.output_path.mkdir(parents=True, exist_ok=True)
            if os.path.exists(args.output_path) and not overwrite_flag:
                metrics_df_old = pd.read_csv(
                    args.output_path / f"metrics_{args.val_dataset}.csv",
                    header=None,
                    names=["Type", "Value"],
                )
                metrics_df = pd.concat([metrics_df_old, metrics_df], ignore_index=True)
            metrics_df.to_csv(
                args.output_path / f"metrics_{args.val_dataset}.csv",
                header=False,
                index=False,
            )
            overwrite_flag = False
            output_data = []
        #metrics_df = metrics_df.round(3)
    return print("运行结束")

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

from attacks.help_function import ownutilities
from attacks.weather import attack_image
from attacks.help_function.weather import get_weather
@torch.enable_grad()
def attack_on_weather_dataloader(    
    args: Namespace,
    attack_args: Dict[str, List[object]],
    model: BaseModel,
    dataloader: DataLoader,
    dataloader_name: str,)-> Dict[str, float]:
    
    metrics_sum = {}

    metrics_individual = None
    if args.write_individual_metrics:
        metrics_individual = {"filename": [], "epe": [], "outlier": []}

    # Run self dataloader
    data_loader, has_gt, has_cam, has_weather = ownutilities.prepare_dataloader(attack_args, shuffle=False, get_weather=True)
    losses = torch.zeros(len(data_loader))
    prev_preds = None
    couter_in = 0
    for i, datachunk in enumerate(tqdm(data_loader)):
        if couter_in ==0:
           for j, inputs in enumerate(tqdm(dataloader)): 
                if args.scale_factor is not None:
                    scale_factor = args.scale_factor
                else:
                    scale_factor = (
                        None
                        if args.max_forward_side is None
                        else float(args.max_forward_side) / min(inputs["images"].shape[-2:])
                        )
                print(inputs)
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
                    attack_args["attack_epsilon"] = attack_args["attack_epsilon"] * 255
                has_ground_truth = True
                targeted_inputs = None
                with torch.no_grad():
                    orig_preds = model(inputs)
                couter_in +=1
                print("couter_in is",couter_in)
        else:
            continue


       
        if has_weather:
            (image1, image2, image1_weather, image2_weather, flow, _, scene_data, extra) = datachunk
        else:
            raise ValueError("Cannot evaluate weather without weather data. Please pass --weather_data to the argument parser.")
        (root,), (split,), (seq,), (base,), (frame,), (weatherdat,) = extra
        weather = get_weather(has_weather, weatherdat, scene_data, attack_args, seed=None, load_only=True)
   
        # scene_data = [i.to(device) for i in scene_data]
        # weather = [i.to(device) for i in weather]
        # image1, image2 = image1.to(device), image2.to(device)
        # flow = flow.to(device)

        #inputs["flows"] = targeted_flow_tensor

        # inputs["scenedata"] = scene_data
        # inputs["weather"] = weather

        targeted_inputs = inputs.copy()

        preds= weather_ds(attack_args, model, targeted_inputs, weather, scene_data) 

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

            if attack_args["attack_targeted"]:
                metrics = model.val_metrics(preds, targeted_inputs)
                metrics_orig_preds = model.val_metrics(preds, orig_preds)
                metrics["val/epe_orig_preds"] = metrics_orig_preds["val/epe"]
                metrics["val/cosim_target"] = torch.mean(cosine_similarity(get_flow_tensors(preds), get_flow_tensors(targeted_inputs)))
                metrics["val/cosim_orig_preds"] = torch.mean(cosine_similarity(get_flow_tensors(preds), get_flow_tensors(orig_preds)))
                if has_ground_truth:
                    metrics_ground_truth = model.val_metrics(preds, inputs)
                    metrics["val/epe_ground_truth"] = metrics_ground_truth["val/epe"]
                    metrics["val/cosim_ground_truth"] = torch.mean(cosine_similarity(get_flow_tensors(preds), get_flow_tensors(inputs)))
            else:
                metrics = model.val_metrics(preds, inputs)
                metrics["val/cosim"] = torch.mean(cosine_similarity(get_flow_tensors(preds), get_flow_tensors(inputs)))

            for k in metrics.keys():
                if metrics_sum.get(k) is None:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += metrics[k].item()
            dataloader.set_postfix(
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

            if attack_args["attack_targeted"]:
                generate_outputs(
                    args,
                    targeted_inputs,
                    preds,
                    dataloader_name,
                    i,
                    targeted_inputs.get("meta"),
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


@torch.enable_grad()
def attack_one_dataloader(
    args: Namespace,
    attack_args: Dict[str, List[object]],
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
                attack_args["attack_epsilon"] = attack_args["attack_epsilon"] * 255

            targeted_inputs = None
            if attack_args["attack_targeted"] or attack_args["attack"] == "pcfa" :
                with torch.no_grad():
                    orig_preds = model(inputs)
                if attack_args["attack_target"] == "negative":
                    targeted_flow_tensor = -orig_preds["flows"]
                else:
                    targeted_flow_tensor = torch.zeros_like(orig_preds["flows"])
                if not "flows" in inputs:
                    inputs["flows"] = targeted_flow_tensor

                targeted_inputs = inputs.copy()
                targeted_inputs["flows"] = targeted_flow_tensor

            # TODO: figure out what to do with scaled images and labels
            # print(attack_args["attack_epsilon"])
            match attack_args["attack"]:  # Commit adversarial attack
                case "fgsm":
                    # inputs["images"] = fgsm(args, inputs, model)
                    images, labels, preds, placeholder = fgsm(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "pgd":
                    # inputs["images"] = cos_pgd(args, inputs, model)
                    images, labels, preds, losses[i] = bim_pgd_cospgd(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "cospgd":
                    # inputs["images"] = cos_pgd(args, inputs, model)
                    images, labels, preds, losses[i] = bim_pgd_cospgd(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "bim":
                    # inputs["images"] = cos_pgd(args, inputs, model)
                    images, labels, preds, losses[i] = bim_pgd_cospgd(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "apgd":
                    # inputs["images"] = fgsm(args, inputs, model)
                    images, labels, preds, placeholder = apgd(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "fab":
                    images, labels, preds, placeholder = fab(
                        attack_args, inputs, model, targeted_inputs
                    )
                case "pcfa":
                    preds, l2_delta1, l2_delta2, l2_delta12 = pcfa(
                        attack_args, model, targeted_inputs
                    )
                # case "weather":

                #     preds= weather_ds(
                #         attack_args, model, targeted_inputs
                #     )   
                case "common_corruptions":
                    preds = common_corrupt(attack_args, inputs, model, args)
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
            if attack_args["attack_targeted"] or attack_args["attack"] == "pcfa":
                metrics = model.val_metrics(preds, targeted_inputs)
                metrics_ground_truth = model.val_metrics(preds, inputs)
                metrics_orig_preds = model.val_metrics(preds, orig_preds)
                metrics["val/epe_ground_truth"] = metrics_ground_truth["val/epe"]
                metrics["val/epe_orig_preds"] = metrics_orig_preds["val/epe"]
            else:
                metrics = model.val_metrics(preds, inputs)

            if attack_args["attack_targeted"] or attack_args["attack"] == "weather":
                metrics = model.val_metrics(preds, targeted_inputs)
                metrics_ground_truth = model.val_metrics(preds, inputs)
                metrics_orig_preds = model.val_metrics(preds, orig_preds)
                metrics["val/epe_ground_truth"] = metrics_ground_truth["val/epe"]
                metrics["val/epe_orig_preds"] = metrics_orig_preds["val/epe"]
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

            if attack_args["attack_targeted"] or attack_args["attack"] == "pcfa" or attack_args["attack"] == "weather":
                generate_outputs(
                    args,
                    targeted_inputs,
                    preds,
                    dataloader_name,
                    i,
                    targeted_inputs.get("meta"),
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
