from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch

from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
# Import cosPGD functions

from attacks.adversarial_attacks_pytorch.torchattacks import APGD
# Attack parameters

def apgd(args: Namespace, inputs: Dict[str, torch.Tensor], model: BaseModel, targeted_inputs: Optional[Dict[str, torch.Tensor]]):
    if args.attack_targeted:
        # TODO: are iterations the same steps?
        attack = APGD(model, eps=args.attack_epsilon, loss=args.attack_loss, verbose=False, steps=args.attack_iterations)
        attack.targeted = True
        attack.set_mode_targeted_by_label()
        perturbed_images = attack(inputs["images"], targeted_inputs["flows"])
    else:
        attack = APGD(model, eps=args.attack_epsilon, loss=args.attack_loss, verbose=False, steps=args.attack_iterations)
        perturbed_images = attack(inputs["images"], inputs["flows"])
    perturbed_inputs = inputs
    perturbed_inputs["images"] = perturbed_images
    preds = model(perturbed_inputs)
    images = None
    labels = None

    return images, labels, preds, None