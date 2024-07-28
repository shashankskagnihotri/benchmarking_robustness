import torch
from torchattacks import FGSM
from typing import Dict, Optional

def perform_fgsm_attack(
    model,
    left: torch.Tensor,
    right: torch.Tensor,
    attack_epsilon: float,
    targeted_disparity: Optional[torch.Tensor] = None,
    targeted: bool = False,
):
    # Ensure the model is in evaluation mode
    model.eval()

    
    # If targeted, set the target, else use the model's own prediction as the target
    if targeted:
        labels = targeted_disparity.unsqueeze(0)
    else:
        with torch.no_grad():
            labels = model(left, right)["disparities"]

    # Instantiate the FGSM attack from torchattacks
    attack = FGSM(model, eps=attack_epsilon)
    
    # If targeted, set the target label for the attack
    if targeted:
        attack.set_targeted_mode()

    # Perform the attack
    inputs = {"left": left, "right": right}
    adv_images = attack(inputs, labels)

    # Prepare the perturbed inputs for the model
    perturbed_inputs = {"images": adv_images}

    # Get predictions on the perturbed images
    preds = model(perturbed_inputs)

    return preds, perturbed_inputs