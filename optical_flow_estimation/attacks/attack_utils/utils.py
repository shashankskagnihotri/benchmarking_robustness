import torch
from typing import Dict
import numpy as np


def get_image_tensors(input_dic: Dict[str, torch.Tensor]):
    image_1 = input_dic["images"][0][0].unsqueeze(0)
    image_2 = input_dic["images"][0][1].unsqueeze(0)
    return image_1, image_2


def get_flow_tensors(input_dic: Dict[str, torch.Tensor]):
    flow = input_dic["flows"][0][0].unsqueeze(0)
    return flow


def get_image_grads(input_dic: Dict[str, torch.Tensor]):
    grad = input_dic["images"].grad
    image_1_grad = grad[0][0].unsqueeze(0)
    image_2_grad = grad[0][1].unsqueeze(0)
    return image_1_grad, image_2_grad


def replace_images_dic(
    input_dic: Dict[str, torch.Tensor],
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    clone: bool = False,
):
    image_pair_tensor = torch.torch.cat((image_1, image_2)).unsqueeze(0)
    if clone:
        output_dic = input_dic.copy()
        output_dic["images"] = image_pair_tensor
        return output_dic
    else:
        input_dic["images"] = image_pair_tensor
        return input_dic


def get_input_format(input):
    if isinstance(input, dict):
        return input
    elif torch.is_tensor(input) and len(input.size()) == 4:
        input_dic = {"images": input.unsqueeze(0)}
        return input_dic
    elif torch.is_tensor(input) and len(input.size()) == 5:
        input_dic = {"images": input}
        return input_dic


# From FlowUnderAttack
def epe(flow1, flow2):
    """ "
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
    diff_squared = (flow1 - flow2) ** 2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=0).sqrt()
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.sum(diff_squared, dim=1).sqrt()
    else:
        raise ValueError(
            "The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: "
            + str(flow1.size())
            + " and "
            + str(flow1.size())
        )
    return epe

def put_things_on_img(img, points, what):
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert len(what.shape) == 2
    assert what.shape[1] == 3
    assert what.shape[0] == points.shape[0]

    h,w,_ = img.shape
    indices = points[:,0] + w * points[:,1]
    indices = indices.unsqueeze(1).repeat(1,3)
    img_flat = img.reshape(-1,3)
    img_flat.scatter_(0, indices, what)
    img2 = img_flat.reshape(h,w,3)
    return img2


def save_weatherfile(file, weather):
  points3D, motion3D, flakes, flakescol, flakestrans = weather
  np.savez(file, points3D=points3D.numpy(), motion3D=motion3D.numpy(), flakes=flakes.numpy(), flakescol=flakescol.numpy(), flakestrans=flakestrans.numpy())


def load_weather(path):
    try:
      weather = np.load(path, allow_pickle=True)
      points3D = weather["points3D"]
      motion3D = weather["motion3D"]
      flakes = weather["flakes"]
      flakescol = weather["flakescol"]
      flakestrans = weather["flakestrans"]
      weather = (points3D, motion3D, flakes, flakescol, flakestrans)
      success = True
    except FileNotFoundError as e:
      print(f"Failed to load weather data from {path}, File Not Found. Please generate weather instead.")
      weather = None
      success = False
    return weather, success
