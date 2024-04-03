import torch
from typing import Dict 

def get_image_tensors(input_dic: Dict[str, torch.Tensor]):
    image_1 = input_dic["images"][0][0].unsqueeze(0)
    image_2 = input_dic["images"][0][1].unsqueeze(0)
    return image_1, image_2

def get_flow_tensors(input_dic: Dict[str, torch.Tensor]):
    flow_1 = input_dic["flows"][0][0].unsqueeze(0)
    flow_2 = input_dic["flows"][0][1].unsqueeze(0)
    return flow_1, flow_2

def get_image_grads(input_dic: Dict[str, torch.Tensor]):
    grad = input_dic["images"].grad
    image_1_grad = grad[0][0].unsqueeze(0)
    image_2_grad = grad[0][1].unsqueeze(0)
    return image_1_grad, image_2_grad

def replace_images_dic(input_dic: Dict[str, torch.Tensor], image_1: torch.Tensor, image_2: torch.Tensor):
    output_dic = input_dic
    image_pair_tensor = torch.torch.cat((image_1, image_2)).unsqueeze(0)
    output_dic["images"] = image_pair_tensor
    return output_dic


