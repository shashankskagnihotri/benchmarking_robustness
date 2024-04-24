from typing import Dict, List
import torch
import numpy as np
from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    replace_images_dic,
)

from imagecorruptions.imagecorruptions import corrupt
import pdb

def common_corrupt(attack_args: Dict[str, List[object]],
                   inputs: Dict[str, torch.Tensor],
                model: BaseModel, args
                ):
    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Convert images to numpy array
    image_1, image_2 = get_image_tensors(inputs)
    pdb.set_trace()
    image_1 = image_1.to(torch.uint8)  # Convert to torch.uint8
    image_2 = image_2.to(torch.uint8)  # Convert to torch.uint8
    image_1_numpy = image_1.cpu().numpy()
    image_2_numpy = image_2.cpu().numpy()

    # Create corruption on each input image
    image_1_corrupt = corrupt(image=image_1_numpy, corruption_name=args.cc_name, severity=args.cc_severity)
    image_2_corrupt = corrupt(image=image_2_numpy, corruption_name=args.cc_name, severity=args.cc_severity)

    # Convert the numpy images back to tensors
    image_1 = torch.from_numpy(image_1_corrupt).to(device)
    image_2 = torch.from_numpy(image_2_corrupt).to(device)

    perturbed_inputs = replace_images_dic(inputs, image_1, image_2, clone=True)

    # Make prediction
    preds = model(perturbed_inputs)

    return preds