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
        attack_iterations = [1,3,4,5] # TODO: check number of iterations
        
        for iteration in range(self.num_iterations):
            perturbed_left.requires_grad = True
            perturbed_right.requires_grad = True
            
            # Forward pass the perturbed images through the model
            outputs = self.model(perturbed_left, perturbed_right)[-1]

            outputs = outputs.cuda()
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
        
            # check if current iteration matches number in iteration list
            if iteration in attack_iterations:
                perturbed_results[iteration]=(perturbed_left, perturbed_right)


        return perturbed_results


class FGSMAttack:
    def __init__(self, model, epsilon, num_iterations,alpha, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.num_iterations = num_iterations
        self.aplha = alpha

    @torch.enable_grad()
    def attack(self, left_image: torch.Tensor, right_image: torch.Tensor, labels: torch.Tensor):
        # Clone original images
        orig_left_image = left_image.clone().detach()
        orig_right_image = right_image.clone().detach()

        # Initialize perturbations for both left and right images
        perturbed_left = left_image.clone().detach()
        perturbed_right = right_image.clone().detach()

        # save perturbed
        perturbed_results = {}
        attack_iterations = [1,3,4,5] # TODO: check number of iterations

        for iteration in range(self.num_iterations):
            # Set the images to require gradient
            perturbed_left.requires_grad = True
            perturbed_right.requires_grad = True

            # Forward pass the perturbed images through the model
            inputs = {"images": [[perturbed_left, perturbed_right]]}
            outputs = self.model(inputs)["disparities"].squeeze(0)

            # Compute the loss
            loss = F.mse_loss(outputs.float(), labels.float())
            if self.targeted:
                loss = -loss

            # Zero all existing gradients
            self.model.zero_grad()

            # Backward pass to compute gradients of the loss w.r.t the perturbed images
            loss.backward()

            # Collect the gradient data
            left_grad = perturbed_left.grad.data
            right_grad = perturbed_right.grad.data

            # Perform the attack step for both left and right images
            perturbed_left = self.fgsm_attack(perturbed_left, left_grad, orig_left_image)
            perturbed_right = self.fgsm_attack(perturbed_right, right_grad, orig_right_image)

            # Detach the perturbed images to avoid accumulating gradients
            perturbed_left = perturbed_left.detach()
            perturbed_right = perturbed_right.detach()

            # check if current iteration matches number in iteration list
            if iteration in attack_iterations:
                perturbed_results[iteration]=(perturbed_left, perturbed_right)
            

        return perturbed_results

    def fgsm_attack(self, perturbed_image: torch.Tensor, data_grad: torch.Tensor, orig_image: torch.Tensor):
        sign_data_grad = data_grad.sign()
        if self.targeted:
            sign_data_grad *= -1

        perturbed_image = perturbed_image.detach() + self.alpha * sign_data_grad

        delta = torch.clamp(perturbed_image - orig_image, min=-self.epsilon, max=self.epsilon)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)

        return perturbed_image

# Based on code / implementaion by jeffkang (https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/pgd.py)
import torch
import torch.nn.functional as F
class PGDAttack:

    def __init__(self, model, epsilon, num_iterations, alpha, random_start=True, targeted=False):
      
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha 
        self.num_iterations = num_iterations
        self.random_start = random_start
        self.targeted = targeted
        self.clamp = (0, 1)

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
        x = torch.clamp(x, *self.clamp)
        return x

    def attack(self, left_image, right_image, labels):
        """
        :param left_image: Left image to perturb
        :param right_image: Right image to perturb
        :param labels: Ground-truth disparity
        :return: Perturbed left and right images
        """
        orig_left_image = left_image.clone().detach()
        orig_right_image = right_image.clone().detach()

        # Initialize perturbations for both left and right images
        perturbed_left = left_image.clone().detach()
        perturbed_right = right_image.clone().detach()

        if self.random_start:
            perturbed_left = self._random_init(perturbed_left)
            perturbed_right = self._random_init(perturbed_right)

        for _ in range(self.num_iterations):
            perturbed_left.requires_grad = True
            perturbed_right.requires_grad = True

            # Forward pass the perturbed images through the model
            inputs = {"images": [[perturbed_left, perturbed_right]]}
            outputs = self.model(inputs)["disparities"].squeeze(0)

            # Compute the loss
            loss = F.mse_loss(outputs.float(), labels.float())
            if self.target:
                loss = -loss

            # Zero all existing gradients
            self.model.zero_grad()

            # Backward pass to compute gradients of the loss w.r.t the perturbed images
            loss.backward()

            # Collect the gradient data
            left_grad = perturbed_left.grad.detach()
            right_grad = perturbed_right.grad.detach()

            # Perform the attack step for both left and right images
            perturbed_left = self.pgd_attack_step(perturbed_left, left_grad, orig_left_image)
            perturbed_right = self.pgd_attack_step(perturbed_right, right_grad, orig_right_image)

            # Detach the perturbed images to avoid accumulating gradients
            perturbed_left = perturbed_left.detach()
            perturbed_right = perturbed_right.detach()

        return perturbed_left, perturbed_right

    def pgd_attack_step(self, perturbed_image, grad, orig_image):
        grad_sign = grad.sign()
        if self.target:
            grad_sign *= -1

        perturbed_image = perturbed_image + self.config['attack_lr'] * grad_sign

        delta = torch.clamp(perturbed_image - orig_image, min=-self.config['eps'], max=self.config['eps'])
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)

        return perturbed_image

    
    def attack(self,):

        for 

    
    def ...attack(self,):
    

    