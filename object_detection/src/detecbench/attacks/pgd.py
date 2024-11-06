from .common import evaluator_process, preprocess_data
from torchvision import transforms
import torch
from mmengine.evaluator import Evaluator
from mmengine.model import BaseModel


def pgd_attack(
    data_batch: dict,
    model: BaseModel,
    steps: int,
    epsilon: float,
    alpha: float,
    targeted: bool,
    random_start: bool,
    evaluators: list[Evaluator],
):
    data_batch_prepro, images = preprocess_data(data_batch, model)
    adv_images = images.clone().float().detach().to("cuda")
    evaluator_process(evaluators[0], data_batch_prepro, model)

    if targeted:
        raise NotImplementedError

    if random_start:
        raise NotImplementedError

    for step in range(steps):
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        model.training = True  # avoid missing arguments error in some models
        losses = model(**data_batch_prepro, mode="loss")
        loss, _ = model.parse_losses(losses)

        model.zero_grad()
        adv_images.grad = None

        loss.backward(retain_graph=True)
        grad = adv_images.grad
        assert grad is not None

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

        data_batch_prepro["inputs"][0] = adv_images
        transforms.Normalize(
            model.data_preprocessor.mean,
            model.data_preprocessor.std,
            inplace=True,
        )(data_batch_prepro["inputs"][0])
        model.training = False  # avoid missing arguments error in some models
        evaluator_process(evaluators[step + 1], data_batch_prepro, model)

    return data_batch_prepro
