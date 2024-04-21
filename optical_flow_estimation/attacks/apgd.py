from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch

from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
# Import cosPGD functions

from attacks.adversarial_attacks_pytorch.torchattacks import APGD
# Attack parameters

def apgd(attack_args: Dict[str, List[object]], inputs: Dict[str, torch.Tensor], model: BaseModel, targeted_inputs: Optional[Dict[str, torch.Tensor]]):
    norm = attack_args["attack_norm"]
    if norm == "inf":
        norm = "Linf"
    elif norm == "two":
        norm = "L2"
    if attack_args["attack_targeted"]:
        # TODO: are iterations the same steps?
        # TODO: implement other iterations
        attack = APGD(model, eps=attack_args["attack_epsilon"], loss=attack_args["attack_loss"], verbose=False, steps=attack_args["attack_iterations"])
        attack.targeted = True
        attack.set_mode_targeted_by_label()
        perturbed_images = attack(inputs["images"], targeted_inputs["flows"])
    else:
        attack = APGD(model, eps=attack_args["attack_epsilon"], loss=attack_args["attack_loss"], verbose=False, steps=attack_args["attack_iterations"])
        perturbed_images = attack(inputs["images"], inputs["flows"])
    perturbed_inputs = inputs
    perturbed_inputs["images"] = perturbed_images
    preds = model(perturbed_inputs)
    images = None
    labels = None

    return images, labels, preds, None