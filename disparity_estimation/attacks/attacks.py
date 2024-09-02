import torch
import torch.nn.functional as F
from typing import Dict, Optional, List



""" Implementation of:
Agnihotri, Shashank, Jung, Steffen, Keuper, Margret. "CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks." 
arXiv preprint arXiv:2302.02213 (2023).

A tool for benchmarking adversarial robustness of pixel-wise prediction tasks.

MIT License

Copyright (c) 2023 Shashank Agnihotri, Steffen Jung, Prof. Dr. Margret Keuper

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

class Attack:
    def __init__(self):
        pass
    
    """
    Function to take one attack step in the l-infinity norm constraint

    perturbed_image: Float tensor of shape [batch size, channels, (image spatial resolution)]
    epsilon: Float tensor: permissible epsilon range
    data_grad: gradient on the image input to the model w.r.t. the loss backpropagated
    orig_image: Float tensor of shape [batch size, channels, (image spatial resolution)]: Original unattacked image, before adding any noise
    alpha: Float tensor: attack step size
    targeted: boolean: Targeted attack or not
    clamp_min: Float tensor: minimum clip value for clipping the perturbed image back to the permisible input space
    clamp_max: Float tensor: maximum clip value for clipping the perturbed image back to the permisible input space
    grad_scale: tensor either single value or of the same shape as data_grad: to scale the added noise
    """
    @staticmethod
    def step_inf(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            targeted,
            clamp_min = 0,
            clamp_max = 1,
            grad_scale = None
        ):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = alpha*data_grad.sign()
        if targeted:
            sign_data_grad *= -1
        if grad_scale is not None:
            sign_data_grad *= grad_scale
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image.detach() + sign_data_grad
        # Adding clipping to maintain [0,1] range
        delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image
    
    
    """
    Function to take one attack step in the l2 norm constraint
    
    perturbed_image: Float tensor of shape [batch size, channels, (image spatial resolution)]
    epsilon: Float tensor: permissible epsilon range
    data_grad: gradient on the image input to the model w.r.t. the loss backpropagated
    orig_image: Float tensor of shape [batch size, channels, (image spatial resolution)]: Original unattacked image, before adding any noise
    alpha: Float tensor: attack step size
    targeted: boolean: Targeted attack or not
    clamp_min: Float tensor: minimum clip value for clipping the perturbed image back to the permisible input space
    clamp_max: Float tensor: maximum clip value for clipping the perturbed image back to the permisible input space
    grad_scale: tensor either single value or of the same shape as data_grad: to scale the added noise
    """
    @staticmethod
    def step_l2(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            targeted,
            clamp_min = 0,
            clamp_max = 1,
            grad_scale = None
        ):
        # normalize gradients
        if targeted:
            data_grad *= -1
        data_grad = Attack.lp_normalize(
            data_grad,
            p = 2,
            epsilon = 1.0,
            decrease_only = False
        )
        if grad_scale is not None:
            data_grad *= grad_scale
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image.detach() + alpha*data_grad
        # clip to l2 ball
        delta = Attack.lp_normalize(
            noise = perturbed_image - orig_image,
            p = 2,
            epsilon = epsilon,
            decrease_only = True
        )
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image
    
    """
    Clamping noise in the l-p norm constraint
    noise: tensor of shape [batch size, (image spatial resolution)]: the noise to be clamped
    p: int: the norm
    epsilon: Float tensor: permissible epsilon range
    decrease_only: boolean: to only clamp the upper bound and not the lower bound
    """
    @staticmethod
    def lp_normalize(
            noise,
            p,
            epsilon = None,
            decrease_only = False
        ):
        if epsilon is None:
            epsilon = torch.tensor(1.0)
        denom = torch.norm(noise, p=p, dim=(-1, -2, -3))
        denom = torch.maximum(denom, torch.tensor(1E-12)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if decrease_only:
            denom = torch.maximum(denom/epsilon, torch.tensor(1))
        else:
            denom = denom / epsilon
        return noise / denom
    
    """
    Initializing noise in the l-infinity norm constraint

    epsilon: Float tensor: permissible epsilon range
    images: Float tensor of shape [batch size, channels, (image spatial resolution)]: Original unattacked image, before adding any noise    
    clamp_min: Float tensor: minimum clip value for clipping the perturbed image back to the permisible input space
    clamp_max: Float tensor: maximum clip value for clipping the perturbed image back to the permisible input space    
    """
    @staticmethod
    def init_linf(
            images,
            epsilon,
            clamp_min = 0,
            clamp_max = 1,
        ):
        noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device)
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images
    
    
    """
    Initializing noise in the l-2 norm constraint

    epsilon: Float tensor: permissible epsilon range
    images: Float tensor of shape [batch size, channels, (image spatial resolution)]: Original unattacked image, before adding any noise    
    clamp_min: Float tensor: minimum clip value for clipping the perturbed image back to the permisible input space
    clamp_max: Float tensor: maximum clip value for clipping the perturbed image back to the permisible input space
    """
    @staticmethod
    def init_l2(
            images,
            epsilon,
            clamp_min = 0,
            clamp_max = 1,
        ):
        noise = torch.FloatTensor(images.shape).uniform_(-1, 1).to(images.device)
        noise = Attack.lp_normalize(
            noise = noise,
            p = 2,
            epsilon = epsilon,
            decrease_only = False
        )
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images
    
    
    """
    Scaling of the pixel-wise loss as proposed by: 
    Gu, Jindong, et al. "Segpgd: An effective and efficient adversarial attack for evaluating and boosting segmentation robustness." 
    European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

    predictions: Float tensor of shape [batch size, channel, (image spatial resolution)]: Predictions made by the model
    labels: The ground truth/target labels, for semantic segmentation index tensor of the shape: [batch size, channel, (image spatial resolution)].
                                     for pixel-wise regression tasks, same shape as predictions
    loss: Float tensor: The loss between the predictions and the ground truth/target
    iteration: Current attack iteration for calculating lambda as used in SegPGD
    iterations: Total number of attack iterations for calculating lambda as used in SegPGD
    targeted: boolean: Targeted attack or not
    """
    @staticmethod
    def segpgd_scale(
            predictions,
            labels,
            loss,
            iteration,
            iterations,
            targeted=False,
        ):
        lambda_t = iteration/(2*iterations)
        output_idx = torch.argmax(predictions, dim=1)
        if targeted:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    lambda_t*loss,
                    (1-lambda_t)*loss
                )
            ) / (predictions.shape[-2]*predictions.shape[-1])
        else:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    (1-lambda_t)*loss,
                    lambda_t*loss
                )
            ) / (predictions.shape[-2]*predictions.shape[-1])
        return loss
    
    
    """
    Scaling of the pixel-wise loss as implemeted by: 
    Agnihotri, Shashank, et al. "CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks." 
    arXiv preprint arXiv:2302.02213 (2023).

    predictions: Float tensor of shape [batch size, channel, (image spatial resolution)]: Predictions made by the model
    labels: The ground truth/target labels, for semantic segmentation index tensor of the shape: [batch size, channel, (image spatial resolution)].
                                     for pixel-wise regression tasks, same shape as predictions
    loss: Float tensor: The loss between the predictions and the ground truth/target
    num_classes: int: For semantic segmentation the number of classes. None for pixel-wise regression tasks
    targeted: boolean: Targeted attack or not
    one_hot: boolean: To use one-hot encoding, SHOULD BE TRUE FOR SEMANTIC SEGMENTATION and FALSE FOR pixel-wise regression tasks
    """
    @staticmethod
    def cospgd_scale(
            predictions,
            labels,
            loss,
            num_classes=None,
            targeted=False,
            one_hot=True,
        ):
        if one_hot:
            transformed_target = torch.nn.functional.one_hot(
                torch.clamp(labels, labels.min(), num_classes-1),
                num_classes = num_classes
            ).permute(0,3,1,2)
        else:
            transformed_target = torch.nn.functional.softmax(labels, dim=1)
        cossim = torch.nn.functional.cosine_similarity(
            torch.nn.functional.softmax(predictions, dim=1),
            transformed_target,
            dim = 1
        )
        if targeted:
            cossim = 1 - cossim # if performing targeted attacks, we want to punish for dissimilarity to the target
        loss = cossim.detach() * loss
        return loss

class CosPGDAttack:
    def __init__(self, model, epsilon, alpha, num_iterations, num_classes=None, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.targeted = targeted
    
    def attack(self, left_image, right_image, labels):
        # Initialize perturbations for both left and right images
        perturbed_left = Attack.init_linf(left_image, self.epsilon)
        perturbed_right = Attack.init_linf(right_image, self.epsilon)

        # save perturbed
        perturbed_results = {}
      
        
        for iteration in range(self.num_iterations):
            perturbed_left.requires_grad = True
            perturbed_right.requires_grad = True
            
            # Forward pass the perturbed images through the model
            outputs = self.model(perturbed_left, perturbed_right)
            print(len(outputs))
            outputs = outputs[-1].cuda()
            labels = labels.cuda()
            
            # Compute the loss
            loss = F.mse_loss(outputs, labels)
            
            # Zero all existing gradients
            self.model.zero_grad()
            
            # Backward pass to compute gradients of the loss w.r.t the perturbed images
            loss.backward()
            
            # Collect the gradient data
            left_grad = perturbed_left.grad.data
            right_grad = perturbed_right.grad.data
            
            # Perform the attack step
            perturbed_left = Attack.step_inf(
                perturbed_image=perturbed_left,
                epsilon=self.epsilon,
                data_grad=left_grad,
                orig_image=left_image,
                alpha=self.alpha,
                targeted=self.targeted,
                clamp_min=0,
                clamp_max=1
            )
            
            perturbed_right = Attack.step_inf(
                perturbed_image=perturbed_right,
                epsilon=self.epsilon,
                data_grad=right_grad,
                orig_image=right_image,
                alpha=self.alpha,
                targeted=self.targeted,
                clamp_min=0,
                clamp_max=1
            )
        
            # save results after every iteration
            perturbed_results[iteration]=(perturbed_left, perturbed_right)


        return perturbed_results


from typing import Dict, List, Optional

class FGSMAttack:
    def __init__(self, model, epsilon, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted

    @torch.enable_grad()
    def attack(self, left_image: torch.Tensor, right_image: torch.Tensor, ground_truth_disparity: torch.Tensor):
        # Klonen der ursprünglichen Bilder
        orig_left_image = left_image.clone().detach()
        orig_right_image = right_image.clone().detach()

        # Initialisierung der Perturbationen für beide Bilder
        perturbed_left = left_image.clone().detach().requires_grad_(True)
        perturbed_right = right_image.clone().detach().requires_grad_(True)

        # Forward Pass: Die perturbierten Bilder durch das Modell leiten
        inputs = {"images": [[perturbed_left, perturbed_right]]}
        predicted_disparity = self.model(inputs)["disparities"].squeeze(0)

        # Berechnung des Verlusts
        loss = F.mse_loss(predicted_disparity.float(), ground_truth_disparity.float())
        if self.targeted:
            loss = -loss

        # Alle bestehenden Gradienten auf null setzen
        self.model.zero_grad()

        # Backward Pass: Gradienten des Verlusts bzgl. der perturbierten Bilder berechnen
        loss.backward()

        # Gradienteninformationen sammeln
        left_grad = perturbed_left.grad.data
        right_grad = perturbed_right.grad.data

        # FGSM-Schritt für beide Bilder durchführen
        perturbed_left = self.fgsm_attack_step(perturbed_left, left_grad, orig_left_image)
        perturbed_right = self.fgsm_attack_step(perturbed_right, right_grad, orig_right_image)

        # Die perturbierten Bilder vom Rechenpfad loslösen, um die Akkumulation von Gradienten zu vermeiden
        perturbed_left = perturbed_left.detach()
        perturbed_right = perturbed_right.detach()

        # Rückgabe der perturbierten Bilder
        return perturbed_left, perturbed_right

    def fgsm_attack_step(self, perturbed_image: torch.Tensor, data_grad: torch.Tensor, orig_image: torch.Tensor):
        # Vorzeichen des Gradienten bestimmen
        sign_data_grad = data_grad.sign()
        if self.targeted:
            sign_data_grad *= -1

        # FGSM-Schritt durchführen
        perturbed_image = perturbed_image.detach() + self.epsilon * sign_data_grad

        # Perturbation beschränken und das Ergebnis auf den erlaubten Wertebereich [0, 1] beschränken
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image
    

    ###

# Based on code / implementaion by jeffkang (https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/pgd.py)

import torch
import torch.nn.functional as F

class PGDAttack:
    def __init__(self, model, epsilon, num_iterations, alpha, norm='inf', random_start=True, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.norm = norm
        self.random_start = random_start
        self.targeted = targeted

    def _random_init(self, x):
        if self.norm == 'inf':
            x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.epsilon
        elif self.norm == 'two':
            x = x + torch.randn_like(x) * self.epsilon
        x = torch.clamp(x, 0, 1)
        return x

    @torch.enable_grad()
    def attack(self, left_image: torch.Tensor, right_image: torch.Tensor, ground_truth_disparity: torch.Tensor):
        # Save the original images
        orig_left_image = left_image.clone().detach()
        orig_right_image = right_image.clone().detach()

        # Start perturbation
        perturbed_left = left_image.clone().detach()
        perturbed_right = right_image.clone().detach()

        if self.random_start:
            perturbed_left = self._random_init(perturbed_left)
            perturbed_right = self._random_init(perturbed_right)

        perturbed_results = {}

        for iteration in range(self.num_iterations):
            perturbed_left.requires_grad = True
            perturbed_right.requires_grad = True

            # Prepare the input for the model
            inputs = {"images": [[perturbed_left, perturbed_right]]}
            predicted_disparity = self.model(inputs)["disparities"].squeeze(0)

            # Calculate the loss between predicted disparity and ground truth
            loss = F.mse_loss(predicted_disparity.float(), ground_truth_disparity.float())
            if self.targeted:
                loss = -loss

            # Backpropagate to obtain gradients
            self.model.zero_grad()
            loss.backward()

            # Update the perturbed images with gradients
            left_grad = perturbed_left.grad.detach()
            right_grad = perturbed_right.grad.detach()

            if self.norm == 'inf':
                perturbed_left = self.pgd_attack_step_inf(perturbed_left, left_grad, orig_left_image)
                perturbed_right = self.pgd_attack_step_inf(perturbed_right, right_grad, orig_right_image)
            elif self.norm == 'two':
                perturbed_left = self.pgd_attack_step_l2(perturbed_left, left_grad, orig_left_image)
                perturbed_right = self.pgd_attack_step_l2(perturbed_right, right_grad, orig_right_image)

            perturbed_left = perturbed_left.detach()
            perturbed_right = perturbed_right.detach()

            # Save the perturbed images after every iteration
            perturbed_results[iteration] = (perturbed_left, perturbed_right)

        return perturbed_results

    def pgd_attack_step_inf(self, perturbed_image: torch.Tensor, grad: torch.Tensor, orig_image: torch.Tensor):
        grad_sign = grad.sign()
        if self.targeted:
            grad_sign *= -1

        # Apply the perturbation step for L∞ norm
        perturbed_image = perturbed_image + self.alpha * grad_sign
        delta = torch.clamp(perturbed_image - orig_image, min=-self.epsilon, max=self.epsilon)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)
        return perturbed_image

    def pgd_attack_step_l2(self, perturbed_image: torch.Tensor, grad: torch.Tensor, orig_image: torch.Tensor):
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
        scaled_grad = grad / (grad_norm + 1e-10)
        if self.targeted:
            scaled_grad *= -1

        # Apply the perturbation step for L2 norm
        perturbed_image = perturbed_image + self.alpha * scaled_grad
        delta = perturbed_image - orig_image
        delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)
        factor = torch.clamp(delta_norm, max=self.epsilon) / (delta_norm + 1e-10)
        perturbed_image = orig_image + delta * factor
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image




# class PGDAttack:
#     def __init__(self, model, epsilon, num_iterations, alpha, random_start=True, targeted=False):
#         self.model = model
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.num_iterations = num_iterations
#         self.random_start = random_start
#         self.targeted = targeted

#     def _random_init(self, x):
#         x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.epsilon
#         x = torch.clamp(x, 0, 1)
#         return x

#     @torch.enable_grad()
#     def attack(self, left_image: torch.Tensor, right_image: torch.Tensor, ground_truth_disparity: torch.Tensor):
#         # Save the original images
#         orig_left_image = left_image.clone().detach()
#         orig_right_image = right_image.clone().detach()

#         # Start perturbation
#         perturbed_left = left_image.clone().detach()
#         perturbed_right = right_image.clone().detach()

#         if self.random_start:
#             perturbed_left = self._random_init(perturbed_left)
#             perturbed_right = self._random_init(perturbed_right)

#         perturbed_results = {}

#         for iteration in range(self.num_iterations):
#             perturbed_left.requires_grad = True
#             perturbed_right.requires_grad = True

#             # Prepare the input for the model
#             inputs = {"images": [[perturbed_left, perturbed_right]]}
#             predicted_disparity = self.model(inputs)["disparities"].squeeze(0)

#             # Calculate the loss between predicted disparity and ground truth
#             loss = F.mse_loss(predicted_disparity.float(), ground_truth_disparity.float())
#             if self.targeted:
#                 loss = -loss

#             # Backpropagate to obtain gradients
#             self.model.zero_grad()
#             loss.backward()

#             # Update the perturbed images with gradients
#             left_grad = perturbed_left.grad.detach()
#             right_grad = perturbed_right.grad.detach()

#             perturbed_left = self.pgd_attack_step(perturbed_left, left_grad, orig_left_image)
#             perturbed_right = self.pgd_attack_step(perturbed_right, right_grad, orig_right_image)

#             perturbed_left = perturbed_left.detach()
#             perturbed_right = perturbed_right.detach()

#             # Save the perturbed images after every iteration
#             perturbed_results[iteration] = (perturbed_left, perturbed_right)

#         return perturbed_results

#     def pgd_attack_step(self, perturbed_image: torch.Tensor, grad: torch.Tensor, orig_image: torch.Tensor):
#         grad_sign = grad.sign()
#         if self.targeted:
#             grad_sign *= -1

#         # Apply the perturbation step
#         perturbed_image = perturbed_image + self.alpha * grad_sign
#         delta = torch.clamp(perturbed_image - orig_image, min=-self.epsilon, max=self.epsilon)
#         perturbed_image = torch.clamp(orig_image + delta, 0, 1)
#         return perturbed_image

''' Based on the implementation of Francesco Croce, Matthias Hein
"Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
ICML 2020

Paper : https://arxiv.org/abs/2003.01690 
https://github.com/fra31/auto-attack/tree/master
implementation can be found here: https://github.com/fra31/auto-attack/blob/master/autoattack/autopgd_base.py
'''
import time
import math
import torch.nn as nn


class APGDAttack():
    def __init__(self, model, num_iterations, norm='Linf', eps=1.0, seed=0, loss='l1', eot_iter=1, rho=.75, verbose=False, device=None):
        self.model = model
        self.num_iterations = num_iterations
        self.eps = eps
        self.norm = norm
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.n_iter_orig =  num_iterations
        self.eps_orig = eps

        if self.norm not in ['Linf', 'L2', 'L1']:
            raise ValueError(f"Unsupported norm: {self.norm}")

        if self.loss not in ['l1', 'l2']:
            raise ValueError(f"Unsupported loss: {self.loss}")

        self.criterion = nn.L1Loss(reduction='none') if self.loss == 'l1' else nn.MSELoss(reduction='none')

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[2:])  # Assuming x is [B, C, H, W] for each image
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        elif self.norm == 'L1':
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        return x / (t.view(-1, *([1] * (self.ndims + 1))) + 1e-12)

    def L1_projection(self, x, delta, eps):
        delta_flat = delta.view(delta.size(0), -1)
        abs_delta = delta_flat.abs()
        abs_delta_sum = abs_delta.sum(dim=1, keepdim=True)
        factor = torch.clamp(abs_delta_sum - eps, min=0) / abs_delta_sum
        delta_flat *= (1 - factor).view(-1, 1)
        return delta_flat.view_as(delta)

    def attack_single_run(self, x_left, x_right, disparity_target=None):
        x_left = x_left.unsqueeze(0) if len(x_left.shape) < self.ndims + 1 else x_left
        x_right = x_right.unsqueeze(0) if len(x_right.shape) < self.ndims + 1 else x_right

        if disparity_target is None:
            disparity_target = self.model(x_left, x_right).detach()

        # Stack the left and right images to create a unified tensor
        x_left_right = torch.stack((x_left, x_right), dim=1)  # [B, 2, C, H, W]

        if self.norm == 'Linf':
            t = 2 * torch.rand(x_left_right.shape).to(self.device).detach() - 1
            x_adv = x_left_right + self.eps * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x_left_right.shape).to(self.device).detach()
            x_adv = x_left_right + self.eps * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x_left_right.shape).to(self.device).detach()
            delta = self.L1_projection(x_left_right, t, self.eps)
            x_adv = x_left_right + t + delta

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        loss_best = self.criterion(self.model(x_best[:, 0], x_best[:, 1]), disparity_target).mean().item()

        step_size = 2. * self.eps / self.num_iterations

        # Initialize a dictionary to store perturbed results for each iteration
        perturbed_results = {}

        for i in range(self.num_iterations):
            x_adv.requires_grad_()
            disparity_pred = self.model(x_adv[:, 0], x_adv[:, 1])
            loss_indiv = self.criterion(disparity_pred, disparity_target).sum()

            grad = torch.autograd.grad(loss_indiv, [x_adv])[0]
            grad_norm = self.normalize(grad)

            if self.norm == 'Linf':
                x_adv = x_adv.detach() + step_size * torch.sign(grad_norm)
            elif self.norm == 'L2':
                x_adv = x_adv.detach() + step_size * grad_norm
            elif self.norm == 'L1':
                grad_topk = grad.abs().view(x_left_right.shape[0], -1).sort(-1)[0]
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv = x_adv.detach() + step_size * sparsegrad.sign() / (grad_norm.sum() + 1e-10)

            x_adv = x_adv.clamp(0., 1.)

            # Update best adversarial example if the loss improved
            loss_curr = self.criterion(self.model(x_adv[:, 0], x_adv[:, 1]), disparity_target).mean().item()
            if loss_curr < loss_best:
                loss_best = loss_curr
                x_best = x_adv.clone()

            # Store the perturbed images for the current iteration
            perturbed_results[i] = (x_adv[:, 0].clone(), x_adv[:, 1].clone())

        return perturbed_results

    def attack(self, x_left, x_right, disparity_target=None):
        self.init_hyperparam(x_left)
        perturbed_results = self.attack_single_run(x_left, x_right, disparity_target)
        return perturbed_results


import torch
import torch.nn.functional as F
from typing import List

class BIMAttack:
  
    def __init__(self, model, epsilon: float, num_iterations: int, alpha: float, norm: str, targeted: bool):
        """see https://arxiv.org/pdf/1607.02533.pdf"""
        self.model = model
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.targeted = targeted
        self.norm = norm

    def _denorm(self, images, mean, std):
        """Denormalize the images"""
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
        return images * std + mean

    def _clip_perturbation(self, adv_images, images):
        """Clip perturbation to be within bounds"""
        if self.norm == "inf":
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
        elif self.norm == "two":
            delta = adv_images - images
            delta_norms = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
            factor = self.epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
        adv_images = torch.clamp(images + delta, 0, 255)
        return adv_images

    @torch.enable_grad()
    def attack(self, left_image: torch.Tensor, right_image: torch.Tensor, labels: torch.Tensor, mean: List[float], std: List[float]):
        orig_left_image = left_image.clone().detach()
        orig_right_image = right_image.clone().detach()

        perturbed_left = left_image.clone().detach()
        perturbed_right = right_image.clone().detach()

        perturbed_results = {}

        for iteration in range(self.num_iterations):
            perturbed_left.requires_grad = True
            perturbed_right.requires_grad = True

            inputs = {"images": [[perturbed_left, perturbed_right]]}
            outputs = self.model(inputs)["disparities"].squeeze(0)
            loss = F.mse_loss(outputs.float(), labels.float())
            if self.targeted:
                loss = -loss

            self.model.zero_grad()
            loss.backward()

            left_grad = perturbed_left.grad.detach()
            right_grad = perturbed_right.grad.detach()

            # Update perturbed images
            if self.targeted:
                left_grad *= -1
                right_grad *= -1
            
            perturbed_left = perturbed_left.detach() + self.alpha * left_grad
            perturbed_right = perturbed_right.detach() + self.alpha * right_grad

            perturbed_left = self._clip_perturbation(perturbed_left, orig_left_image)
            perturbed_right = self._clip_perturbation(perturbed_right, orig_right_image)

            perturbed_left = perturbed_left.detach()
            perturbed_right = perturbed_right.detach()

            perturbed_results[iteration] = (perturbed_left, perturbed_right)

        return perturbed_results
