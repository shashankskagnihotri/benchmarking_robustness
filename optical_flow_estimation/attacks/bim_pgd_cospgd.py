from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch

from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attack_utils.utils import get_image_tensors, get_image_grads, replace_images_dic, get_flow_tensors
# Import cosPGD functions
from cospgd import functions as attack_functions
import torch.nn as nn

criterion = nn.MSELoss(reduction="none")

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
    loss = epe(pred_flows.float(), labels.float())
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
        loss = criterion(pred_flows.float(), labels.float())

    loss = loss.mean()
    return perturbed_images, labels, preds, loss.item()


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
