import torch
from mmengine.evaluator import Evaluator
from mmengine.model import BaseModel
from torchvision import transforms

from .common import Attack


class BIM(Attack):
    """"""

    def __init__(
        self,
        epsilon: float,
        alpha: float,
        steps: int,
        targeted: bool = False,
        norm: str = "inf",
    ):
        """""" ""
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.targeted = targeted
        self.norm = norm

    def run_batch(self, data_batch: dict, model: BaseModel, evaluators: list[Evaluator]):
        """"""
        data_batch_prepro, images = self._preprocess_data(data_batch, model)
        adv_images = images.clone().float().detach().to("cuda")
        self._evaluator_process(evaluators[0], data_batch_prepro, model)

        for step in range(self.steps):
            # Get gradient
            adv_images.requires_grad = True
            data_batch_prepro["inputs"][0] = adv_images
            model.training = True  # avoid missing arguments error in some models
            losses = model(**data_batch_prepro, mode="loss")
            cost, _ = model.parse_losses(losses)
            model.zero_grad()
            cost.backward(retain_graph=True)
            grad = adv_images.grad
            assert grad is not None

            # Collect the element-wise sign of the data gradient
            sign_data_grad = grad.sign()

            # Create the perturbed image by adjusting each pixel of the input image
            if self.targeted:
                sign_data_grad *= -1
            adv_images = adv_images.detach() + self.alpha * sign_data_grad

            # Adding clipping to maintain [0,255] range
            if self.norm == "inf":
                delta = torch.clamp(adv_images - images, min=-1 * self.epsilon, max=self.epsilon)
            elif self.norm == "two":
                delta = adv_images - images
                batch_size = images.shape[0]
                delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                factor = self.epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)
            else:
                raise NotImplementedError
            adv_images = torch.clamp(images + delta, 0, 255).detach()

            data_batch_prepro["inputs"][0] = adv_images
            transforms.Normalize(
                model.data_preprocessor.mean,
                model.data_preprocessor.std,
                inplace=True,
            )(data_batch_prepro["inputs"][0])
            model.training = False  # avoid missing arguments error in some models
            self._evaluator_process(evaluators[step + 1], data_batch_prepro, model)

        return data_batch_prepro
