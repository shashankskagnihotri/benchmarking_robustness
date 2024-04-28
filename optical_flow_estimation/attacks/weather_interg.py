from __future__ import print_function
import mlflow
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os

from tqdm import tqdm
from mlflow import log_metric, log_param

import sys
sys.path.append('/pfs/work7/workspace/scratch/ma_xinygao-team_project_fss2024/benchmarking_robustness/optical_flow_estimation')

from DistractingDownpour.helper_functions import ownutilities
from DistractingDownpour.weather_attack.render import render
from DistractingDownpour.weather_attack.weather import get_weather, recolor_weather

# from torch.cuda.amp import GradScaler, autocast

# 检查CUDA是否可用
print("是否存在cuda:", torch.cuda.is_available())

batch_size = 1

import torch.optim as optim
import torch.cuda.amp as amp

from argparse import  Namespace
from typing import Any, Dict, List, Optional
import torch
from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import get_image_tensors, get_image_grads, replace_images_dic, get_flow_tensors
import torch.nn as nn
import attacks.attack_utils.loss_criterion as losses
import pdb

def calc_metrics_adv(flow_pred, target, flow_pred_init):

    epe_pred_target = losses.avg_epe(flow_pred, target)
    epe_pred_pred_init = losses.avg_epe(flow_pred, flow_pred_init)

    # print("avg ADV-EPE 结果是: %.12f, avg ADV-EPE tgt    : %.12f" % (epe_pred_pred_init,epe_pred_target))

    return epe_pred_target, epe_pred_pred_init

from DistractingDownpour.helper_functions import datasets
from DistractingDownpour.helper_functions.config_specs import Paths, Conf
from torch.utils.data import DataLoader, Subset


def weather_ds(attack_args: Dict[str, List[object]], model: BaseModel, targeted_inputs: Dict[str, torch.Tensor]):
    
    print("打印所有的键：", attack_args)
    """
    Performs a weather attack on a given model and for all images of a specified dataset.
    """
        # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    print("现在正在运行的是：", device)
    # =======
 
    data_loader, has_gt, has_cam, has_weather = ownutilities.prepare_dataloader(attack_args, shuffle=False, get_weather=True)

    for batch, datachunck in enumerate(tqdm(data_loader)):

        (image1, image2, image1_weather, image2_weather, flow, _, scene_data, extra) = datachunck

        (root,), (split,), (seq,), (base,), (frame,), (weatherdat,) = extra

        weather = get_weather(has_weather, weatherdat, scene_data, attack_args, seed=None, load_only=True)
        
        scene_data = [i.to(device) for i in scene_data]
        weather = [i.to(device) for i in weather]
        image1, image2 = image1.to(device), image2.to(device)
        flow = flow.to(device)
        preds, aee_adv_gt = weather_attack(model, targeted_inputs, device, weather, scene_data, attack_args)




    # ====================

    # has_weather=True
    # weatherdat = attack_args["weatherdat"]

    #  # 定义一个空列表来存储处理后的天气数据
    # weather_data = []
    # # 定义两个空列表来存储每次调用 weather_attack 返回的 preds 和 aee_adv_gt
    # all_preds = []
    # all_aee_adv_gt = []

    # def process_files_in_directory(directory):
    #     """
    #     Process all files in a directory recursively.
    #     """
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             if file.endswith(".npz"):
    #                 file_path = os.path.join(root, file)
    #                 # 调用 get_weather 函数并处理图片
    #                 weather = get_weather(has_weather, file_path, attack_args, load_only=True)
    #                 # 调用 weather_attack 函数并获取返回值
    #                 preds, aee_adv_gt = weather_attack(model, targeted_inputs, device, weather, scene_data, attack_args)
    #                 # 将返回值添加到相应的列表中
    #                 all_preds.append(preds)
    #                 all_aee_adv_gt.append(aee_adv_gt)
    #     return 
    # # 调用函数并传入需要遍历的文件夹路径
    # process_files_in_directory(weatherdat)


    print("所有 preds 值是：", all_preds)
    print("所有 aee_adv_gt 值是：", all_aee_adv_gt)

    return preds, aee_adv_gt





from attacks.attack_utils.attack_args_parser import AttackArgumentParser

# Assuming `args` is your argument parser object

# 删 ,scene_data
def weather_attack(model, targeted_inputs, device, weather, attack_args: Dict[str, List[object]]):
   
    print("天气weather输入为：",weather )
    torch.autograd.set_detect_anomaly(False)

    # print("目标输入为：",targeted_inputs )
    image1, image2 = get_image_tensors(targeted_inputs)
    image1, image2 = image1.to(device), image2.to(device)
    
    eps_box = 1e-7

    full_P, ext1, rel_mat, gt1_depth, gt2_depth = scene_data
    
    flow = get_flow_tensors(targeted_inputs)
    flow = flow.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    # gt1_depth.requires_grad = False
    # gt2_depth.requires_grad = False
    # full_P.requires_grad = False
    # ext1.requires_grad = False
    # rel_mat.requires_grad = False

    scene_data = full_P, ext1, rel_mat, gt1_depth, gt2_depth
    initpos, motion, flakes, flakes_color, flakes_transp = weather

    print("Saving initial weather")
    image1_weather_init, image2_weather_init = render(image1.detach().clone(), image2.detach().clone(), scene_data, weather, args)
       
       # ===== ATTACK OPTIMIZATION =====

    torch.autograd.set_detect_anomaly(False)

    # prepare and rescale images
    offsets = torch.zeros_like(initpos).detach().clone()
    motion_offsets = torch.zeros_like(initpos)

    flow_weather_init = None
    flakes_transp_init = flakes_transp.clone().detach()
    flakes_color_init = flakes_color.clone().detach()
    offset_init = offsets.detach().clone()

    # non-learnable parameters
    initpos.requires_grad = False
    flakes.requires_grad = False
    motion.requires_grad = False
    flakes_color.requires_grad = False
    flakes_color_inf = torch.atanh( 2. * (1.- eps_box) * (flakes_color) - (1 - eps_box)  ) # switch color to [-infty, infty] for better optimization
    # Note: the transparencies are already atanh-transformed
    print("初始flake_color是：",flakes_color_inf)

    # learnable parameters
    offsets.requires_grad = True
    motion_offsets.requires_grad = True
    flakes_transp.requires_grad = True
    flakes_color_inf.requires_grad = True


    # if attack_args["learn_offset"]:
    #     learnparams += [offsets]
    # if attack_args["learn_motionoffset"]:
    #     learnparams += [motion_offsets]
    # if attack_args["learn_transparency"]:
    #     learnparams += [flakes_transp]
    # if attack_args["learn_color"]:
    #     learnparams += [flakes_color_inf]
    # if learnparams == []:
    #     raise ValueError("No learnable parameters were passed in the argument parser. Cannot optimize particles.")
    #optimizer = optim.LBFGS(learnparams, optimizer_lr=0.001)
                # Predict the flow 预测光流 -220
    flakes_color_img = (1./2.) * 1. / (1. - eps_box) * (torch.tanh(flakes_color_inf) + (1 - eps_box) )
    weather = (initpos+offsets, motion+motion_offsets, flakes, flakes_color_img, flakes_transp)
    print("运行过程中weather：",weather)    

    flow_weather = replace_images_dic(targeted_inputs, image1, image2, clone=True) #perturbed_inputs
    print("运行过程中flow_weather：",flow_weather)  
    # preds = model(flow_weather)

    from attacks.helpmodel_for_weather import ScaledInputWeatherModel 
    preds = ScaledInputWeatherModel.forward(image1, image2, weather=weather, scene_data=scene_data, args_=attack_args, test_mode=True)

    # flow_weather = replace_images_dic(targeted_inputs, image1, image2, clone=True) #perturbed_inputs
    flow_weather = preds


    # flow_weather = flow_weather.to(device) #同保留 flow_pred
    flow_weather = flow_weather.to(device) #同保留 flow_pred

    # define the initial flow, the target, and update mu 同保留 -230
    flow_weather_init = flow_weather.detach().clone()
    flow_weather_init.requires_grad = False

    # flow_init = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, test_mode=True)
    # [flow_init] = ownutilities.postprocess_flow(args.net, padder, flow_init)
    # 疑问
    # flow_init = replace_images_dic(targeted_inputs, image1, image2, clone=True) 
    # flow_init = flow_init.to(device).detach()

    # define target (potentially based on first flow prediction) 定义攻击目标 -235
    # define attack target
    target = get_flow_tensors(targeted_inputs)
    target = target.to(device)
    target.requires_grad = False

    # initialize values and best values
    # EPE statistics for the unattacked flow

    # Zero all existing gradients 保留 同-252
    model.zero_grad()
    #optimizer.zero_grad()

    for steps in range(attack_args["weather_steps"]):

        # Calculate loss
    
        loss = losses.loss_weather(flow_weather, target, f_type=attack_args["attack_loss"], init_pos=initpos, offsets=offsets, motion_offsets=motion_offsets, flakes_transp=flakes_transp, flakes_transp_init=flakes_transp_init, alph_offsets=1000, alph_motion=1000, alph_transp=0)
 
        loss.backward()

        flakes_color_img = (1./2.) * 1. / (1. - eps_box) * (torch.tanh(flakes_color_inf) + (1 - eps_box) )
        weather = (initpos+offsets, motion+motion_offsets, flakes, flakes_color_img, flakes_transp)
        print("循环中weather：", weather)

        # flow_weather = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, test_mode=True, weather=weather, scene_data=scene_data, args_=args)
        # [flow_weather] = ownutilities.postprocess_flow(args.net, padder, flow_weather)

        flow_weather = replace_images_dic(targeted_inputs, image1, image2, clone=True)
        preds = model(flow_weather, weather= weather)
        print("循环中preds：", preds)
        flow_weather = preds["flows"].squeeze(0) #flow_pred
        flow_weather = flow_weather.to(device)

        aee_adv_tgt, aee_adv_pred = calc_metrics_adv(flow_weather, target, flow_weather_init)
        aee_adv_gt = losses.avg_epe(flow_weather, flow)
        print("计算结果aee：", aee_adv_gt)
    # return aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt_init, aee_adv_tgt_init, aee_adv_pred_init, aee_adv_gt_min_val, aee_adv_tgt_min_val, aee_adv_pred_min_val
    return preds, aee_adv_gt