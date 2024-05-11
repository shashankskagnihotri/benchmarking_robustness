from argparse import Namespace
from typing import Any, Dict, List, Optional
import torch
from ptlflow_attacked.ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    get_image_grads,
    replace_images_dic,
    get_flow_tensors,
)
import torch.nn as nn
import attacks.attack_utils.loss_criterion as losses
import pdb
import torch.optim as optim


def pcfa(
    attack_args: Dict[str, List[object]],
    model: BaseModel,
    targeted_inputs: Dict[str, torch.Tensor],
):
    """
    Performs an PCFA attack on a given model and for all images of a specified dataset.
    """

    optim_mu = 2500.0 / attack_args["attack_epsilon"]
    if attack_args["attack_target"] not in ["zero"]:
            optim_mu = 1.5*optim_mu

    eps_box = attack_args["attack_alpha"]

    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Make sure the model is not trained:
    for param in model.parameters():
        param.requires_grad = False

    preds, l2_delta1, l2_delta2, l2_delta12 = pcfa_attack(
        model, targeted_inputs, eps_box, device, optim_mu, attack_args
    )

    return preds #, l2_delta1, l2_delta2, l2_delta12


def pcfa_attack(model, targeted_inputs, eps_box, device, optim_mu, attack_args):
    """Subroutine to optimize a PCFA perturbation on a given image pair. For a specified number of steps.

    Args:
        model (torch.nn.module):
            a pytorch model which is set to eval and which is implemented in ownutilities.['preprocess_img','compute_flow','postprocess_flow']
        image1 (torch.tensor):
            image 1 of a scene with dimensions (b,3,h,w)
        image2 (torch.tensor):
            image 2 of a scene with dimensions (b,3,h,w)
        flow (torch.tensor):
            intial (unattacked) flow field (resulting from img1 and img2) which can be used to log the current effect induced by the patch (same spatial dimension as images!)
        batch (int):
            current image counter in the enumeration of the test set
        distortion_folder (string):
            name for a folder that will be created to hold data and visualizations of the distortions that are trained during the PCFA
        eps_box (float):
            relaxation of the box constraint due to numerical reasons (try 1e-07)
        device (torch.device):
            Select the device for the images.
        optimizer_lr (float):
            optimizer learning rate (try 5e-03)
        has_gt (boolean):
            is the ground truth known
        optim_mu (float):
            regularization parameter of the constraint in the unconstraing optimization (try 5e05)
        args (Namespace):
            command line arguments


    Returns:
        aee_gt (float):
            Average Endpoint Error of the ground truth w.r.t. zero flow (none if has_gt is false)
        aee_tgt (float):
            Average Endpoint Error of the target flow w.r.t. zero flow
        aee_gt_tgt (float):
            Average Endpoint Error of the ground truth towards the target flow (none if has_gt is false)
        aee_adv_tgt (float):
            Average Endpoint Error of the adversarial predicted flow (after the perturbation is added) towards the target flow
        aee_adv_pred (float):
            Average Endpoint Error of the adversarial predicted flow (after the perturbation is added) towards the initially predicted flow
        l2_delta2 (float):
            l2 error of the preturbation on image1
        l2_delta2 (float):
            l2 error of the preturbation on image2
        l2_delta12 (float):
            scalar average L2-norm of two perturbations
        aee_adv_tgt_min_val (float):
            minimal (best attack) Average Endpoint Error of the adversarial predicted flow towards the target flow, while also not violating the constraint, i.e. delta12 < args.delta_bound
        aee_adv_pred_min_val (float):
            best attack Average Endpoint Error of the adversarial predicted flow towards the initially predicted flow, while also not violating the constraint
        delta12_min_val (float):
            scalar average L2-norm of two perturbations of the best attack

    Extended Output:
        Files in .npy and .png format of initial images and flow, perturbations, and the adversarial images and flow are saved.
        The best attack perturbation images and flows are labeled with "*best*" and correspond to the values of "*_min*".
    """
    torch.autograd.set_detect_anomaly(True)

    model = InputModel(model, eps_box, variable_change=attack_args["pcfa_boxconstraint"] in ["change_of_variables"])

    image1, image2 = get_image_tensors(targeted_inputs)
    image1, image2 = image1.to(device), image2.to(device)
    # pdb.set_trace()
    flow = get_flow_tensors(targeted_inputs)
    flow = flow.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False

    # initialize perturbation and auxiliary variables:
    delta1 = torch.zeros_like(image1)
    delta2 = torch.zeros_like(image2)
    delta1 = delta1.to(device)
    delta2 = delta2.to(device)

    nw_input1 = None
    nw_input2 = None

    flow_pred_init = None

    # Set up the optimizer and variables if individual perturbations delta1 and delta2 for images 1 and 2 should be trained
    delta1.requires_grad = False
    delta2.requires_grad = False

    if attack_args["pcfa_boxconstraint"] in ["change_of_variables"]:
        nw_input1 = torch.atanh(
            2.0 * (1.0 - eps_box) * (image1 + delta1) - (1 - eps_box)
        )
        nw_input2 = torch.atanh(
            2.0 * (1.0 - eps_box) * (image2 + delta2) - (1 - eps_box)
        )
    else:
        nw_input1 = image1 + delta1
        nw_input2 = image2 + delta2

    nw_input1.requires_grad = True
    nw_input2.requires_grad = True

    optimizer = optim.LBFGS([nw_input1, nw_input2], max_iter=10)

    # Predict the flow
    preds = model(nw_input1, nw_input2)
    flow_pred = preds["flows"].squeeze(0)
    flow_pred = flow_pred.to(device)

    # define the initial flow, the target, and update mu
    flow_pred_init = flow_pred.detach().clone()
    flow_pred_init.requires_grad = False

    # define target (potentially based on first flow prediction)
    target = get_flow_tensors(targeted_inputs)
    target = target.to(device)
    target.requires_grad = False
    
    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    l2_delta1, l2_delta2, l2_delta12 = 0, 0, 0

    for steps in range(attack_args["attack_iterations"]):
        # Calculate the deltas from the quantities that go into the network
        delta1, delta2 = extract_deltas(
            nw_input1,
            nw_input2,
            image1,
            image2,
            attack_args["pcfa_boxconstraint"],
            eps_box=eps_box,
        )
        # Calculate the loss
        loss = losses.loss_delta_constraint(
            flow_pred,
            target,
            delta1,
            delta2,
            device,
            delta_bound=attack_args["attack_epsilon"],
            mu=optim_mu,
            f_type=attack_args["attack_loss"],
        )
        # Update the optimization parameters
        loss.backward()

        def closure():
            optimizer.zero_grad()

            # Predict the flow
            flow_closure = model(nw_input1, nw_input2)["flows"].squeeze(0)
            flow_closure = flow_closure.to(device)

            delta1_closure, delta2_closure = extract_deltas(
                nw_input1,
                nw_input2,
                image1,
                image2,
                attack_args["pcfa_boxconstraint"],
                eps_box=eps_box,
            )
            loss_closure = losses.loss_delta_constraint(
                flow_closure,
                target,
                delta1_closure,
                delta2_closure,
                device,
                delta_bound=attack_args["attack_epsilon"],
                mu=optim_mu,
                f_type=attack_args["attack_loss"],
            )
            loss_closure.backward()

            return loss_closure

        # Update the optimization parameters
        optimizer.step(closure)

        # calculate the magnitude of the updated distortion
        delta1, delta2 = extract_deltas(
            nw_input1,
            nw_input2,
            image1,
            image2,
            attack_args["pcfa_boxconstraint"],
            eps_box=eps_box,
        )
        # The nw_inputs remain unchanged in this case, and can be directly fed into the network again for further perturbation training

        # Re-predict flow with the perturbed image, and update the flow prediction for the next iteration
        preds = model(nw_input1, nw_input2)
        flow_pred = preds["flows"].squeeze(0)
        flow_pred = flow_pred.to(device)

        l2_delta1 = torchfloat_to_float64(losses.two_norm_avg(delta1))
        l2_delta2 = torchfloat_to_float64(losses.two_norm_avg(delta2))
        l2_delta12 = torchfloat_to_float64(losses.two_norm_avg_delta(delta1, delta2))

    return preds, l2_delta1, l2_delta2, l2_delta12


# For PCFA
def extract_deltas(nw_input1, nw_input2, image1, image2, boxconstraint, eps_box=0.0):

    if boxconstraint in ["change_of_variables"]:
        delta1 = (1.0 / 2.0) * 1.0 / (1.0 - eps_box) * (
            torch.tanh(nw_input1) + (1.0 - eps_box)
        ) - image1
        delta2 = (1.0 / 2.0) * 1.0 / (1.0 - eps_box) * (
            torch.tanh(nw_input2) + (1.0 - eps_box)
        ) - image2
    else:  # case clipping (ScaledInputModel also treats everything that is not change_of_variables by clipping it into range before feeding it to network.)
        delta1 = torch.clamp(nw_input1, 0.0, 1.0) - image1
        delta2 = torch.clamp(nw_input2, 0.0, 1.0) - image2

    return delta1, delta2


def torchfloat_to_float64(torch_float):
    """helper function to convert a torch.float to numpy float

    Args:
        torch_float (torch.float):
            scalar floating point number in torch

    Returns:
        numpy.float: floating point number in numpy
    """
    float_val = float(torch_float.detach().cpu().numpy())
    return float_val



class InputModel(nn.Module):
    def __init__(self, model, eps_box, variable_change=False,  **kwargs):
        """
        Initializes a Model with rescaled input requirements. It calls a pretrained model that takes inputs in [0,255], but the ScaledInputModel version of it can handle inputs in [0,1] and transforms them before passing them to the pretrained model.
        This might be necessary for attacks that work on different paramter spaces, or with substitued variables
        Args:
            net: a string describing the network type
            make_unit_input (bool): if True, the Model assumes that it receives inputs in [0,1], but that the pretrained model requires them in [0,255]
            variable_change (bool): if True, the Model assumes that it will be given a variable as required for the change of variable proposed in the Carlini&Wagner attack.
            **kwargs: Futher arguments for ownutilities.import_and_load()
        """
        super(InputModel, self).__init__()

        self.var_change = variable_change

        self.eps_box = eps_box

        self.model_loaded = model


    def forward(self, image1, image2):
        """
        Performs a forward pass of the pretrained model as specified in self.model_loaded.
        Optionally, it transforms the input image1 and image2 previous to feeding them into the model.
        When specifying delta1 or delta2, one or both of these tensors are added as perturbations to the input images.
        If only "delta1" is specified, this tensor is added to both images and the result is cliped to the allowed image range [0,1].
        If make_unit_input=True was specified for the ScaledInputModel, images1 and 2 are assumed to be in [0,1] when they enter forward, but will be rescaled to [0,255] before being passed to the pretrained model.
        If variable_change=True was specified for the ScaledInputModel, images1 and 2 are assumed to be not the image information, but the w-variable from the Carlini&Wagner model. Hence they are transformed into their image representations, before being fed to to the model.

        Args:
            image1 (tensor): first input image for model in [0,1] with dimensions (b,3,H,W)
            image2 (tensor): second input image for model in [0,1] with dimensions (b,3,H,W)
            *args: additional arguments for model
            delta1 (optional, tensor): optional distortion to be added to image1 in [0,1] with dimensions (b,3,H,W)
            delta2 (optional, tensor): optional distortion to be added to image2 in [0,1] with dimensions (b,3,H,W)
            **kwargs: additional kwargs for model

        Returns:
            tensor: returns the output of a forward pass of the pytorch.model on image1 and image2, after the specified transformations to images1 and 2.
        """

        # Perform the Carlini&Wagner Change of Variables, if the ScaledInputModel was configured to do so.
        if self.var_change:
            image1 = (1./2.) * 1. / (1. - self.eps_box) * (torch.tanh(image1) + (1 - self.eps_box) )
            image2 = (1./2.) * 1. / (1. - self.eps_box) * (torch.tanh(image2) + (1 - self.eps_box) )


        # Clipping case, which will only clip something if change of variables was not defined. otherwise, the change of variables has already brought the iamges into the range [0,1]
        image1 = torch.clamp(image1, 0., 1.)
        image2 = torch.clamp(image2, 0., 1.)

        image_pair_tensor = torch.torch.cat((image1, image2)).unsqueeze(0)
        dic = {"images": image_pair_tensor}

        return self.model_loaded(dic)