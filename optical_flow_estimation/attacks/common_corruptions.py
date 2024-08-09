from typing import Dict, List
import torch
import numpy as np
from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    replace_images_dic,
)
from imagecorruptions import corrupt


def common_corrupt(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
):
    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Convert images to numpy array
    image_1, image_2 = get_image_tensors(inputs)

     # Check initial shapes
    print(f"Original shapes: image_1: {image_1.shape}, image_2: {image_2.shape}")

    # 确保图像通道数为 3，如果是 RGBA 图像则裁剪通道
    if image_1.shape[1] == 4:  # 通道数为 4，可能是 RGBA 图像
        image_1 = image_1[:, :3, :, :]  # 只保留前三个通道
    if image_2.shape[1] == 4:  # 通道数为 4，可能是 RGBA 图像
        image_2 = image_2[:, :3, :, :]

    # Rearrange the dimensions of the tensor, remove the batch dimension
    image_1_permuted = image_1.squeeze(0).permute(1, 2, 0)  # To RGB format (height x width x channels)
    image_2_permuted = image_2.squeeze(0).permute(1, 2, 0)

    # Check shapes after permutation
    print(f"Permuted shapes: image_1: {image_1_permuted.shape}, image_2: {image_2_permuted.shape}")

    # Create the image as a numpy array scaled to [0, 255]
    image_1_numpy = (image_1_permuted * 255).clamp(0, 255).byte().cpu().numpy()
    image_2_numpy = (image_2_permuted * 255).clamp(0, 255).byte().cpu().numpy()

    # Create corruption on each input image
    image_1_corrupt = corrupt(
        image_1_numpy,
        corruption_name=attack_args["cc_name"],
        severity=attack_args["cc_severity"],
    )
    image_2_corrupt = corrupt(
        image_2_numpy,
        corruption_name=attack_args["cc_name"],
        severity=attack_args["cc_severity"],
    )

    # Rescale the numpy images to [0, 1] and convert them back to tensors
    image_1 = torch.from_numpy(image_1_corrupt / 255).to(device).float()
    image_2 = torch.from_numpy(image_2_corrupt / 255).to(device).float()

    # 确保图像通道数为 3
    if image_1.shape[1] == 4:
        image_1 = image_1[:, :3, :, :]  # 只保留前三个通道
    if image_2.shape[1] == 4:
        image_2 = image_2[:, :3, :, :]

    # 确保张量的维度
    if image_1.dim() == 3:
        image_1 = image_1.unsqueeze(0)  # 添加批量维度 (1 x channels x height x width)
    if image_2.dim() == 3:
        image_2 = image_2.unsqueeze(0)  # 添加批量维度 (1 x channels x height x width)

    # # Rearrange the dimensions of the tensor back to the original shape
    # image_1 = image_1.permute(2, 0, 1).unsqueeze(0)  # To RGB format (batch_size x channels x height x width)
    # image_2 = image_2.permute(2, 0, 1).unsqueeze(0)

    # 调整张量的维度
    image_1 = image_1.permute(0, 2, 3, 1)  # 转换为 (batch_size x height x width x channels)
    image_2 = image_2.permute(0, 2, 3, 1)

    # Check shapes after conversion back to tensor
    print(f"Tensors shapes: image_1: {image_1.shape}, image_2: {image_2.shape}")

    perturbed_inputs = replace_images_dic(inputs, image_1, image_2, clone=True)
    perturbed_inputs["images"] = torch.cat((image_1, image_2), dim=0)

     # Check final shape
    print(f"Concatenated images shape: {perturbed_inputs['images'].shape}")

    # Make prediction
    preds = model(perturbed_inputs)

    return preds, perturbed_inputs
