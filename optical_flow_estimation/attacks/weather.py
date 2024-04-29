#from __future__ import print_function
from argparse import Namespace
from typing import Any, Dict, List, Optional

from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    get_image_grads,
    replace_images_dic,
    get_flow_tensors,
)

from attacks.help_function import ownutilities,  parsing_file, targets, logging
# from attacks.help_function.config_specs import Conf
from attacks.help_function.render import render
from attacks.help_function.weather import get_weather, recolor_weather
# from attacks.help_function.utils import load_weather
import torch
import torch.nn as nn
import attacks.attack_utils.loss_criterion as losses
import pdb
import torch.optim as optim
import mlflow
import numpy as np
import torch.nn.functional as F


from tqdm import tqdm
from mlflow import log_metric, log_param

def get_optimizer(optimizer_name, optimization_parameters, optimizer_lr=0.001):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(optimization_parameters, lr=optimizer_lr)
    else:
        raise ValueError("The selected optimizer option '%s' is unknown. Select 'Adam'." )
    return optimizer

def weather_ds(
    attack_args: Dict[str, List[object]],
    model: BaseModel,
    targeted_inputs: Dict[str, torch.Tensor],
    ):
    print("打印所有键",attack_args)

     # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    print("现在正在运行的是：", device)
    
    data_loader, has_gt, has_cam, has_weather = ownutilities.prepare_dataloader(args=attack_args, shuffle=False, get_weather=True)
    
    for batch, datachunck in enumerate(tqdm(data_loader)):

            if has_weather:
              (image1, image2, image1_weather, image2_weather, flow, _, scene_data, extra) = datachunck
            else:
              raise ValueError("Cannot evaluate weather without weather data. Please pass --weather_data to the argument parser.")

            (root,), (split,), (seq,), (base,), (frame,), (weatherdat,) = extra
            weather = get_weather(has_weather, weatherdat, scene_data, args=attack_args, seed=None, load_only=True)

            # print("scene_data有什么：", weather)
            # print("weather有什么：", weather)
            scene_data = [i.to(device) for i in scene_data]
            weather = [i.to(device) for i in weather]
            image1, image2 = image1.to(device), image2.to(device)
            flow = flow.to(device)

            preds = attack_image(model, targeted_inputs, attack_args, device, scene_data, weather)
            print("打印preds", preds)
    # has_weather=True
    # weather = get_weather(has_weather, weatherdat, scene_data, attack_args, seed=None, load_only=True)
        
    # scene_data = [i.to(device) for i in scene_data]
    # weather = [i.to(device) for i in weather]
    
    # preds = attack_image(model, targeted_inputs, device, weather, scene_data, attack_args)

    return preds


def attack_image(model, targeted_inputs, attack_args, device, scene_data, weather):

    torch.autograd.set_detect_anomaly(True)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    #print("字典是",targeted_inputs)
    image1, image2 = get_image_tensors(targeted_inputs)
    # print("image1是：",image1)
    # print("image2是：",image2)
    # print("device是：",device)
    image1, image2 = image1.to(device), image2.to(device)
     
        

    # pdb.set_trace()
    flow = get_flow_tensors(targeted_inputs)
    flow = flow.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    
    eps_box = 1e-7
    full_P, ext1, rel_mat, gt1_depth, gt2_depth = scene_data

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    gt1_depth.requires_grad = False
    gt2_depth.requires_grad = False
    full_P.requires_grad = False
    ext1.requires_grad = False
    rel_mat.requires_grad = False
    
    scene_data = full_P, ext1, rel_mat, gt1_depth, gt2_depth
    # print("full_P有什么：", full_P) #RIGHT!
    # print("ext1有什么：", ext1) #RIGHT!

    # print("scene_data有什么：", weather)
    # print("weather有什么：", weather)
    # print(len(weather))
    initpos, motion, flakes, flakes_color, flakes_transp = weather
    

     # ===== ATTACK OPTIMIZATION =====

    torch.autograd.set_detect_anomaly(True)

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

     # learnable parameters
    offsets.requires_grad = True
    motion_offsets.requires_grad = True
    flakes_transp.requires_grad = True
    flakes_color_inf.requires_grad = True

    learnparams = []
    if attack_args["weather_learn_offset"]:
        learnparams += [offsets]
    if attack_args["weather_learn_motionoffset"]:
        learnparams += [motion_offsets]
    if  attack_args["weather_learn_transparency"]:
        learnparams += [flakes_transp]
    if attack_args["weather_learn_color"]:
        learnparams += [flakes_color_inf]
    if learnparams == []:
        raise ValueError("No learnable parameters were passed in the argument parser. Cannot optimize particles.")
    
    optimizer = get_optimizer(attack_args["weather_optimizer"], learnparams, optimizer_lr=0.001)

    # Predict the flow
    flakes_color_img = (1./2.) * 1. / (1. - eps_box) * (torch.tanh(flakes_color_inf) + (1 - eps_box) )
    weather = (initpos+offsets, motion+motion_offsets, flakes, flakes_color_img, flakes_transp)
    

    rendered_image1, rendered_image2 = render(image1, image2, scene_data, weather, args=attack_args)
    # print("image1是：",image1)
    # print("image2是：",image2)
    # print("rendered_image1是：",rendered_image1)
    # print("rendered_image1是：",rendered_image2)
    perturbed_inputs = replace_images_dic(
        targeted_inputs, rendered_image1, rendered_image2, clone=True
    )
    preds = model(perturbed_inputs)
    flow_weather_pred = preds["flows"].squeeze(0)
    flow_weather_pred = flow_weather_pred.to(device)
    flow_weather_init = flow_weather_pred.detach().clone()
    flow_weather_init.requires_grad = False
    # from DistractingDownpour.helper_functions.own_models import ScaledInputWeatherModel
    # # 创建 ScaledInputWeatherModel 的实例
    # modelw = ScaledInputWeatherModel(args=attack_args, model="GMA", make_unit_input=False)
    # pred_flow = modelw.forward(rendered_image1, rendered_image2, weather=weather, scene_data=scene_data, args_=attack_args, test_mode=True)
    # perturbed_inputs = replace_images_dic(targeted_inputs, rendered_image1, rendered_image2, clone=True)
    # flow_weather_pred = pred_flow["flows"].squeeze(0)  #flow_pred =...
    # flow_weather_pred = flow_weather_pred.to(device)
    # define the initial flow, the target, and update mu
    # flow_weather_init = flow_weather_pred.detach().clone()
    # flow_weather_init.requires_grad = False

    rendered_image1_init, rendered_image2_init = render(image1, image2, scene_data, weather, args=attack_args)
    perturbed_inputs_init = replace_images_dic(
        targeted_inputs, rendered_image1_init, rendered_image2_init, clone=True
    )
    flow_init = model(perturbed_inputs_init)
    targeted_inputs_init = flow_init["flows"].squeeze(0)
    targeted_inputs_init = targeted_inputs_init.to(device).detach()

    
    # flow_init = ScaledInputWeatherModel.forward(rendered_image1_init, rendered_image2_init, weather=None, scene_data=None, args_=None, test_mode=True)
    # perturbed_inputs_init = replace_images_dic(targeted_inputs, rendered_image1, rendered_image2, clone=True)
    # targeted_inputs_init = flow_init["flows"].squeeze(0) 
    # targeted_inputs_init = targeted_inputs_init.to(device).detach()

    # define target (potentially based on first flow prediction)
    # define attack target
    # target = targets.get_target(attack_args["weather_target"], flow_init, device=device)
    # target = target.to(device)
    # target.requires_grad = False
    target = get_flow_tensors(targeted_inputs)
    target = target.to(device)
    target.requires_grad = False


    flow_weather_min = flow_weather_init.clone().detach()
    offsets_min = offsets.clone().detach()
    motion_offsets_min = motion_offsets.clone().detach()
    flakes_color_min = flakes_color_img.clone().detach()
    flakes_transp_min = flakes_transp.clone().detach()

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    for steps in range(attack_args["weather_steps"]):
        print("into Schleife")
        # Calculate loss
        loss = losses.loss_weather(flow_weather_pred, target, f_type=attack_args["attack_loss"], init_pos=initpos, offsets=offsets, motion_offsets=motion_offsets, flakes_transp=flakes_transp, flakes_transp_init=flakes_transp_init, alph_offsets=attack_args["weather_alph_motion"], alph_motion=attack_args["weather_alph_motionoffset"], alph_transp=0)
 
        #loss.backward()

        if attack_args["weather_optimizer"] in ['Adam']:
            optimizer.step()
        else:
            raise RuntimeWarning('Unknown optimizer, no optimization step was performed')

        flakes_color_img = (1./2.) * 1. / (1. - eps_box) * (torch.tanh(flakes_color_inf) + (1 - eps_box) )
        weather = (initpos+offsets, motion+motion_offsets, flakes, flakes_color_img, flakes_transp)


        # flow_weather = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, weather=weather, scene_data=scene_data, test_mode=True, args_=args)
        # [flow_weather] = ownutilities.postprocess_flow(args.net, padder, flow_weather)
        # flow_weather = flow_weather.to(device)

        # pred_flow = ScaledInputWeatherModel.forward(rendered_image1, rendered_image2, weather=weather, scene_data=scene_data, args_=None, test_mode=True)
        # perturbed_inputs = replace_images_dic(targeted_inputs, rendered_image1, rendered_image2, clone=True)
        # flow_weather_pred = pred_flow["flows"].squeeze(0)  #flow_pred =...
        # flow_weather_pred = flow_weather_pred.to(device)

        pred_flow = model(perturbed_inputs)
        flow_weather_pred = pred_flow["flows"].squeeze(0)
        flow_weather_pred = flow_weather_pred.to(device)
        print("PRINT",flow_weather_pred)
    loss.backward()
    return pred_flow

