from .common import evaluator_process, preprocess_data
from torchvision import transforms
import torch
from mmengine.evaluator import Evaluator
from mmengine.model import BaseModel

def fgsm_attack(
    data_batch: dict,
    model: BaseModel,
    epsilon: float,
    alpha: float,
    norm: str,
    targeted: bool,
    evaluators: list[Evaluator],
):
    data_batch_prepro, images = preprocess_data(data_batch, model)
    adv_images = images.clone().float().detach().to("cuda")

    evaluator_process(evaluators[0], data_batch_prepro, model)

    # Get gradient
    adv_images.requires_grad = True
    data_batch_prepro["inputs"][0] = adv_images

    model.training = True  # avoid missing arguments error in some models
    losses = model(**data_batch_prepro, mode="loss")

    loss, _ = model.parse_losses(losses)
    model.zero_grad()
    loss.backward(retain_graph=True)
    grad = adv_images.grad
    assert grad is not None

    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    if targeted:
        sign_data_grad *= -1
    adv_images = adv_images.detach() + alpha * sign_data_grad

    if norm == "inf":
        delta = torch.clamp(adv_images - images, min=-1 * epsilon, max=epsilon)
    elif norm == "two":
        delta = adv_images - images
        batch_size = images.shape[0]
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
    else:
        raise NotImplementedError

    # Clipping to maintain [0, 255] range
    adv_images = torch.clamp(images + delta, 0, 255).detach()

    # Return the perturbed image
    adv_images.requires_grad = False
    data_batch_prepro["inputs"][0] = adv_images
    transforms.Normalize(
        model.data_preprocessor.mean,
        model.data_preprocessor.std,
        inplace=True,
    )(data_batch_prepro["inputs"][0])
    model.training = False  # avoid missing arguments error in some models
    evaluator_process(evaluators[1], data_batch_prepro, model)

    return data_batch_prepro
