from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch

from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import get_image_tensors, get_image_grads, replace_images_dic, get_flow_tensors
# Import cosPGD functions
from cospgd import functions as attack_functions
import torch.nn as nn
from attacks.attack_utils.loss_criterion import LossCriterion

@torch.enable_grad()
def bim_pgd_cospgd(args: Namespace, inputs: Dict[str, torch.Tensor], model: BaseModel, targeted_inputs: Optional[Dict[str, torch.Tensor]]):
    """Perform bim, pgd or cospgd adversarial attack on input images.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    inputs : Dict[str, torch.Tensor]
        Input images and labels.
    model : BaseModel
        The model for adversarial attack.

    Returns
    -------
    torch.Tensor
        Perturbed images.
    """

    if args.attack_targeted:
        labels = targeted_inputs["flows"].squeeze(0)
    else:
        labels = inputs["flows"].squeeze(0)

    criterion = LossCriterion(args.attack_loss)

    orig_image_1 = inputs["images"][0].clone()[0].unsqueeze(0)
    orig_image_2 = inputs["images"][0].clone()[1].unsqueeze(0)

    image_1, image_2 = get_image_tensors(inputs)
    
    if 'pgd' in args.attack:
        if args.attack_norm == "inf":
            image_1 = attack_functions.init_linf(
                                image_1,
                                epsilon = args.attack_epsilon,
                                clamp_min = 0,
                                clamp_max = 1
                            )
            image_2 = attack_functions.init_linf(
                                image_2,
                                epsilon = args.attack_epsilon,
                                clamp_min = 0,
                                clamp_max = 1
                            )
        elif args.attack_norm == "two":
            image_1 = attack_functions.init_l2(
                                image_1,
                                epsilon = args.attack_epsilon,
                                clamp_min = 0,
                                clamp_max = 1
                            )
            image_2 = attack_functions.init_l2(
                                image_2,
                                epsilon = args.attack_epsilon,
                                clamp_min = 0,
                                clamp_max = 1
                            )
    else:
        args.attack_alpha = args.attack_epsilon
    
    perturbed_inputs = replace_images_dic(inputs, image_1, image_2)
    perturbed_images = perturbed_inputs["images"]
    perturbed_images.requires_grad=True
    perturbed_inputs["images"] = perturbed_images

    preds = model(perturbed_inputs)
    pred_flows = preds["flows"].squeeze(0)
    
    # loss = criterion(pred_flows.float(), labels.float())
    loss = criterion.loss(pred_flows.float(), labels.float())
    for t in range(args.attack_iterations):
        if args.attack == "cospgd":
            loss = attack_functions.cospgd_scale(
                                predictions = pred_flows,
                                labels = labels.float(),
                                loss = loss,
                                targeted = args.attack_targeted,
                                one_hot = False
                            )
        loss = loss.mean()
        loss.backward()
        image_1_adv, image_2_adv = get_image_tensors(perturbed_inputs)
        image_1_grad, image_2_grad = get_image_grads(perturbed_inputs)
        if args.attack_norm == 'inf':
            image_1_adv = attack_functions.step_inf(
                perturbed_image = image_1_adv,
                epsilon = args.attack_epsilon,
                data_grad = image_1_grad,
                orig_image = orig_image_1,
                alpha = args.attack_alpha,
                targeted = args.attack_targeted,
                clamp_min = 0,
                clamp_max = 1,
                grad_scale = None
            )
            image_2_adv = attack_functions.step_inf(
                perturbed_image = image_2_adv,
                epsilon = args.attack_epsilon,
                data_grad = image_2_grad,
                orig_image = orig_image_2,
                alpha = args.attack_alpha,
                targeted = args.attack_targeted,
                clamp_min = 0,
                clamp_max = 1,
                grad_scale = None
            )
        elif args.attack_norm == 'two':
            image_1_adv = attack_functions.step_l2(
                perturbed_image = image_1_adv,
                epsilon = args.attack_epsilon,
                data_grad = image_1_grad,
                orig_image = orig_image_1,
                alpha = args.attack_alpha,
                targeted = args.attack_targeted,
                clamp_min = 0,
                clamp_max = 1,
                grad_scale = None
            )
            image_2_adv = attack_functions.step_l2(
                perturbed_image = image_2_adv,
                epsilon = args.attack_epsilon,
                data_grad = image_2_grad,
                orig_image = orig_image_2,
                alpha = args.attack_alpha,
                targeted = args.attack_targeted,
                clamp_min = 0,
                clamp_max = 1,
                grad_scale = None
            )

        perturbed_inputs = replace_images_dic(perturbed_inputs, image_1_adv, image_2_adv)
        perturbed_images = perturbed_inputs["images"]
        perturbed_images.requires_grad=True
        perturbed_inputs["images"] = perturbed_images

        preds = model(perturbed_inputs)
        pred_flows = preds["flows"].squeeze(0)
        loss = criterion.loss(pred_flows.float(), labels.float())

    loss = loss.mean()
    return perturbed_images, labels, preds, loss.item()
