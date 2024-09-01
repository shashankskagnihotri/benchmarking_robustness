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

# class FGSMAttack:
#     def __init__(self, model, epsilon, num_iterations,alpha, targeted=False):
#         self.model = model
#         self.epsilon = epsilon
#         self.targeted = targeted
#         self.num_iterations = num_iterations
#         self.alpha = alpha

#     @torch.enable_grad()
#     def attack(self, left_image: torch.Tensor, right_image: torch.Tensor, labels: torch.Tensor):
#         # Clone original images
#         orig_left_image = left_image.clone().detach()
#         orig_right_image = right_image.clone().detach()

#         # Initialize perturbations for both left and right images
#         perturbed_left = left_image.clone().detach().requires_grad_(True)
#         perturbed_right = right_image.clone().detach().requires_grad_(True)

#         # save perturbed
#         perturbed_results = {}
       

#         for iteration in range(self.num_iterations):
          
#             # Forward pass the perturbed images through the model
#             inputs = {"images": [[perturbed_left, perturbed_right]]}
#             outputs = self.model(inputs)["disparities"].squeeze(0)

#             # Compute the loss
#             loss = F.mse_loss(outputs.float(), labels.float())
#             if self.targeted:
#                 loss = -loss

#             # Zero all existing gradients
#             self.model.zero_grad()

#             # Backward pass to compute gradients of the loss w.r.t the perturbed images
#             loss.backward()

#             # Collect the gradient data
#             left_grad = perturbed_left.grad.data
#             right_grad = perturbed_right.grad.data

#             # Perform the attack step for both left and right images
#             perturbed_left = self.fgsm_attack(perturbed_left, left_grad, orig_left_image)
#             perturbed_right = self.fgsm_attack(perturbed_right, right_grad, orig_right_image)

#             # Detach the perturbed images to avoid accumulating gradients
#             perturbed_left = perturbed_left.detach()
#             perturbed_right = perturbed_right.detach()

       
#             # save results after every iteration
#             perturbed_results[iteration]=(perturbed_left, perturbed_right)
            

#         return perturbed_results

#     def fgsm_attack(self, perturbed_image: torch.Tensor, data_grad: torch.Tensor, orig_image: torch.Tensor):
#         sign_data_grad = data_grad.sign()
#         if self.targeted:
#             sign_data_grad *= -1

#         perturbed_image = perturbed_image.detach() + self.alpha * sign_data_grad

#         delta = torch.clamp(perturbed_image - orig_image, min=-self.epsilon, max=self.epsilon)
#         perturbed_image = torch.clamp(orig_image + delta, 0, 1)

#         return perturbed_image

# Based on code / implementaion by jeffkang (https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/pgd.py)

class PGDAttack:
    def __init__(self, model, epsilon, num_iterations, alpha, random_start=True, targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.random_start = random_start
        self.targeted = targeted

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.epsilon
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

            perturbed_left = self.pgd_attack_step(perturbed_left, left_grad, orig_left_image)
            perturbed_right = self.pgd_attack_step(perturbed_right, right_grad, orig_right_image)

            perturbed_left = perturbed_left.detach()
            perturbed_right = perturbed_right.detach()

            # Save the perturbed images after every iteration
            perturbed_results[iteration] = (perturbed_left, perturbed_right)

        return perturbed_results

    def pgd_attack_step(self, perturbed_image: torch.Tensor, grad: torch.Tensor, orig_image: torch.Tensor):
        grad_sign = grad.sign()
        if self.targeted:
            grad_sign *= -1

        # Apply the perturbation step
        perturbed_image = perturbed_image + self.alpha * grad_sign
        delta = torch.clamp(perturbed_image - orig_image, min=-self.epsilon, max=self.epsilon)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)
        return perturbed_image

'''Implementation of Francesco Croce, Matthias Hein
Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
ICML 2020
https://arxiv.org/abs/2003.01690 Francesco Croce
https://github.com/fra31/auto-attack/tree/master
https://github.com/fra31/auto-attack/blob/master/autoattack/autopgd_base.py
'''
import time
import math



# class APGDAttack:
#     def __init__(self, predict, n_iter=100, norm='Linf', n_restarts=1, eps=None, seed=0, loss='ce', eot_iter=1, rho=.75, topk=None, verbose=False, device=None, use_largereps=False, is_tf_model=False, logger=None):
#         self.model = predict
#         self.n_iter = n_iter
#         self.eps = eps
#         self.norm = norm
#         self.n_restarts = n_restarts
#         self.seed = seed
#         self.loss = loss
#         self.eot_iter = eot_iter
#         self.thr_decr = rho
#         self.topk = topk
#         self.verbose = verbose
#         self.device = device
#         self.use_largereps = use_largereps
#         self.is_tf_model = is_tf_model
#         self.y_target = None
#         self.logger = logger

#         assert self.norm in ['Linf', 'L2', 'L1']
#         assert self.eps is not None

#         self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
#         self.n_iter_min = max(int(0.06 * self.n_iter), 1)
#         self.size_decr = max(int(0.03 * self.n_iter), 1)

#     def init_hyperparam(self, x):
#         if self.device is None:
#             self.device = x.device
#         self.orig_dim = list(x.shape[1:])
#         self.ndims = len(self.orig_dim)
#         if self.seed is None:
#             self.seed = time.time()

#     def L1_projection(self, x2, y2, eps1):
#         x = x2.clone().float().view(x2.shape[0], -1)
#         y = y2.clone().float().view(x2.shape[0], -1)
#         sigma = y.clone().sign()
#         u = torch.min(1 - x - y, x + y)
#         u = torch.min(torch.zeros_like(y), u)
#         l = -torch.clone(y).abs()
#         d = u.clone()

#         bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
#         bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

#         inu = 2*(indbs < u.shape[1]).float() - 1
#         size1 = inu.cumsum(dim=1)

#         s1 = -u.sum(dim=1)

#         c = eps1 - y.clone().abs().sum(dim=1)
#         c5 = s1 + c < 0
#         c2 = c5.nonzero().squeeze(1)

#         s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

#         if c2.nelement() != 0:
#             lb = torch.zeros_like(c2).float()
#             ub = torch.ones_like(lb) * (bs.shape[1] - 1)
#             nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
#             counter2 = torch.zeros_like(lb).long()
#             counter = 0

#             while counter < nitermax:
#                 counter4 = torch.floor((lb + ub) / 2.)
#                 counter2 = counter4.type(torch.LongTensor)

#                 c8 = s[c2, counter2] + c[c2] < 0
#                 ind3 = c8.nonzero().squeeze(1)
#                 ind32 = (~c8).nonzero().squeeze(1)

#                 if ind3.nelement() != 0:
#                     lb[ind3] = counter4[ind3]
#                 if ind32.nelement() != 0:
#                     ub[ind32] = counter4[ind32]

#                 counter += 1

#             lb2 = lb.long()
#             alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
#             d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

#         return (sigma * d).view(x2.shape)

#     def attack(self, x, y=None, best_loss=False, x_init=None):
#         assert self.loss in ['ce', 'dlr']
#         if not y is None and len(y.shape) == 0:
#             x.unsqueeze_(0)
#             y.unsqueeze_(0)
#         self.init_hyperparam(x)

#         x = x.detach().clone().float().to(self.device)
#         if not self.is_tf_model:
#             y_pred = self.model(x).max(1)[1]
#         else:
#             y_pred = self.model.predict(x).max(1)[1]
#         if y is None:
#             y = y_pred.detach().clone().long().to(self.device)
#         else:
#             y = y.detach().clone().long().to(self.device)

#         adv = x.clone()
#         if self.loss != 'ce-targeted':
#             acc = y_pred == y
#         else:
#             acc = y_pred != y
#         loss = -1e10 * torch.ones_like(acc).float()
#         if self.verbose:
#             print('-------------------------- ', 'running {}-attack with epsilon {:.5f}'.format(self.norm, self.eps), '--------------------------')
#             print('initial accuracy: {:.2%}'.format(acc.float().mean()))

#         if self.use_largereps:
#             epss = [3. * self.eps, 2. * self.eps, 1. * self.eps]
#             iters = [.3 * self.n_iter, .3 * self.n_iter, .4 * self.n_iter]
#             iters = [math.ceil(c) for c in iters]
#             iters[-1] = self.n_iter - sum(iters[:-1])
#             if self.verbose:
#                 print('using schedule [{}x{}]'.format('+'.join([str(c) for c in epss]), '+'.join([str(c) for c in iters])))

#         startt = time.time()
#         if not best_loss:
#             torch.random.manual_seed(self.seed)
#             torch.cuda.random.manual_seed(self.seed)

#             for counter in range(self.n_restarts):
#                 ind_to_fool = acc.nonzero().squeeze()
#                 if len(ind_to_fool.shape) == 0:
#                     ind_to_fool = ind_to_fool.unsqueeze(0)
#                 if ind_to_fool.numel() != 0:
#                     x_to_fool = x[ind_to_fool].clone()
#                     y_to_fool = y[ind_to_fool].clone()

#                     if not self.use_largereps:
#                         res_curr = self.attack_single_run(x_to_fool, y_to_fool)
#                     else:
#                         res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
#                     best_curr, acc_curr, loss_curr, adv_curr = res_curr
#                     ind_curr = (acc_curr == 0).nonzero().squeeze()

#                     acc[ind_to_fool[ind_curr]] = 0
#                     adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
#                     if self.verbose:
#                         print('restart {} - robust accuracy: {:.2%}'.format(counter, acc.float().mean()), '- cum. time: {:.1f} s'.format(time.time() - startt))

#             return adv

#         else:
#             adv_best = x.detach().clone()
#             loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
#             for counter in range(self.n_restarts):
#                 best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
#                 ind_curr = (loss_curr > loss_best).nonzero().squeeze()
#                 adv_best[ind_curr] = best_curr[ind_curr] + 0.
#                 loss_best[ind_curr] = loss_curr[ind_curr] + 0.

#                 if self.verbose:
#                     print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))

#             return adv_best

#     def autopgd_attack(self, x, y, x_init=None):
#         assert self.loss in ['ce', 'dlr']
#         if len(x.shape) < self.ndims:
#             x = x.unsqueeze(0)
#             y = y.unsqueeze(0)
#         self.init_hyperparam(x)

#         x_adv = x.clone()
#         if self.norm == 'Linf':
#             t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
#             x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
#         elif self.norm == 'L2':
#             t = torch.randn(x.shape).to(self.device).detach()
#             x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
#         elif self.norm == 'L1':
#             t = torch.randn(x.shape).to(self.device).detach()
#             x_adv = x + self.eps * torch.ones_like(x).detach() * self.L1_projection(t, x, self.eps)

#         adv = self.attack(x_adv, y, best_loss=False, x_init=x)
#         return adv

#     def decr_eps_pgd(self, x, y, epss, iters):
#         assert len(epss) == len(iters)
#         assert len(epss) > 1
#         adv = x.clone()
#         if self.loss == 'ce':
#             loss_best = torch.zeros(x.shape[0]).to(x.device)
#         else:
#             loss_best = torch.ones(x.shape[0]).to(x.device) * (-float('inf'))
#         for eps, n_iter in zip(epss, iters):
#             if self.verbose:
#                 print('eps = {:.5f} - iter = {}'.format(eps, n_iter))
#             self.eps = eps
#             if not self.use_largereps:
#                 x_adv = self.attack_single_run(x, y, n_iter)
#             else:
#                 x_adv = self.attack_single_run(x, y, n_iter, use_pgd=False)
#             if self.loss == 'ce':
#                 loss_curr = self.model(x_adv).max(1)[0] - loss_best
#                 loss_best = torch.max(loss_best, loss_curr)
#             else:
#                 loss_curr = self.model(x_adv).max(1)[0]
#                 loss_best = torch.max(loss_best, loss_curr)
#             adv = x_adv
#         return adv

#     def attack_single_run(self, x, y, n_iter=None, use_pgd=True):
#         if n_iter is None:
#             n_iter = self.n_iter

#         x_adv = x.clone().detach().to(self.device)
#         adv_final = x_adv.clone().detach()
#         acc = (self.model(x_adv).max(1)[1] == y)
#         for _ in range(self.eot_iter):
#             adv_curr, acc_curr = self.iterate(x_adv, y, n_iter)
#             acc = acc * acc_curr
#             adv_final[~acc] = adv_curr[~acc]
#         return adv_final, acc, self.model(adv_final).max(1)[0], adv_final

#     def iterate(self, x_adv, y, n_iter):
#         # This function should contain your iterative attack algorithm
#         # Example:
#         for _ in range(n_iter):
#             x_adv = self.step(x_adv, y)
#         return x_adv, self.model(x_adv).max(1)[1] == y

#     def step(self, x_adv, y):
#         # This function should contain one step of your iterative attack algorithm
#         # Example:
#         return x_adv  # Placeholder, implement your attack step here

#     def normalize(self, x):
#         if self.norm == 'L2':
#             norm = torch.norm(x.view(x.shape[0], -1), dim=1, keepdim=True)
#             norm[norm == 0] = 1
#             return x / norm.view(x.shape[0], 1, 1, 1)
#         elif self.norm == 'L1':
#             norm = x.view(x.shape[0], -1).abs().sum(dim=1, keepdim=True)
#             norm[norm == 0] = 1
#             return x / norm.view(x.shape[0], 1, 1, 1)
#         else:
#             return x

#     def adjust_eps(self, epsilon):
#         if self.verbose:
#             print(f'Adjusting epsilon to {epsilon}')
#         self.eps = epsilon

import torch.nn as nn




class APGDAttack():
    def __init__(self, model, n_iter=100, norm='L2', eps=1.0, seed=0, loss='l1', eot_iter=1, rho=.75, verbose=False, device=None):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.n_iter_orig = n_iter
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

        step_size = 2. * self.eps / self.n_iter

        # Initialize a dictionary to store perturbed results for each iteration
        perturbed_results = {}

        for i in range(self.n_iter):
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



# class APGDAttack():
#     def __init__(self, model, n_iter=100, norm='L2', eps=1.0, seed=0, loss='l2', eot_iter=1, rho=.75, verbose=False, device=None):
#         self.model = model
#         self.n_iter = n_iter
#         self.eps = eps
#         self.norm = norm
#         self.seed = seed
#         self.loss = loss
#         self.eot_iter = eot_iter
#         self.thr_decr = rho
#         self.verbose = verbose
#         self.device = device
#         self.use_rs = True
#         self.n_iter_orig = n_iter
#         self.eps_orig = eps

#         if self.norm not in ['Linf', 'L2', 'L1']:
#             raise ValueError(f"Unsupported norm: {self.norm}")

#         if self.loss not in ['l1', 'l2']:
#             raise ValueError(f"Unsupported loss: {self.loss}")

#         self.criterion = nn.L1Loss(reduction='none') if self.loss == 'l1' else nn.MSELoss(reduction='none')

#     def init_hyperparam(self, x):
#         if self.device is None:
#             self.device = x.device
#         self.orig_dim = list(x.shape[1:])
#         self.ndims = len(self.orig_dim)
#         if self.seed is None:
#             self.seed = time.time()

#     def normalize(self, x):
#         if self.norm == 'Linf':
#             t = x.abs().view(x.shape[0], -1).max(1)[0]
#         elif self.norm == 'L2':
#             t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
#         elif self.norm == 'L1':
#             t = x.abs().view(x.shape[0], -1).sum(dim=-1)
#         return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

#     def attack_single_run(self, x, x_init=None):
#         x = x.unsqueeze(0) if len(x.shape) < self.ndims else x

#         if self.norm == 'Linf':
#             t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
#             x_adv = x + self.eps * self.normalize(t)
#         elif self.norm == 'L2':
#             t = torch.randn(x.shape).to(self.device).detach()
#             x_adv = x + self.eps * self.normalize(t)
#         elif self.norm == 'L1':
#             t = torch.randn(x.shape).to(self.device).detach()
#             delta = self.L1_projection(x, t, self.eps)
#             x_adv = x + t + delta

#         x_adv = x_adv.clamp(0., 1.)
#         x_best = x_adv.clone()
#         loss_best = self.criterion(self.model(x_best), self.model(x)).mean().item()

#         step_size = 2. * self.eps / self.n_iter

#         for i in range(self.n_iter):
#             x_adv.requires_grad_()
#             logits = self.model(x_adv)
#             loss_indiv = self.criterion(logits, self.model(x)).sum()

#             grad = torch.autograd.grad(loss_indiv, [x_adv])[0]
#             grad_norm = self.normalize(grad)

#             if self.norm == 'Linf':
#                 x_adv = x_adv.detach() + step_size * torch.sign(grad_norm)
#             elif self.norm == 'L2':
#                 x_adv = x_adv.detach() + step_size * grad_norm
#             elif self.norm == 'L1':
#                 grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
#                 sparsegrad = grad * (grad.abs() >= grad_topk).float()
#                 x_adv = x_adv.detach() + step_size * sparsegrad.sign() / (grad_norm.sum() + 1e-10)
            
#             x_adv = x_adv.clamp(0., 1.)

#             # Update best adversarial example if the loss improved
#             loss_curr = self.criterion(self.model(x_adv), self.model(x)).mean().item()
#             if loss_curr < loss_best:
#                 loss_best = loss_curr
#                 x_best = x_adv.clone()

#         return x_best

#     def perturb(self, x):
#         self.init_hyperparam(x)
#         adv = self.attack_single_run(x)
#         return adv

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
