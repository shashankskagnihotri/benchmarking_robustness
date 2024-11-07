import torch
from mmengine.evaluator import Evaluator
from mmengine.model import BaseModel
from torchvision import transforms

from .common import Attack


class PGD(Attack):
    """"""

    def __init__(
        self,
        epsilon: float = 8,
        alpha: float = 0.01 * 255,
        steps: int = 20,
        targeted: bool = False,
        random_start: bool = False,
    ):
        """"""
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.targeted = targeted
        self.random_start = random_start

    def run_batch(self, data_batch: dict, model: BaseModel, evaluators: list[Evaluator]):
        """"""

        data_batch_prepro, images = self._preprocess_data(data_batch, model)
        adv_images = images.clone().float().detach().to("cuda")
        self._evaluator_process(evaluators[0], data_batch_prepro, model)

        if self.targeted:
            raise NotImplementedError

        if self.random_start:
            raise NotImplementedError

        for step in range(self.steps):
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

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=255).detach()

            data_batch_prepro["inputs"][0] = adv_images
            transforms.Normalize(
                model.data_preprocessor.mean,
                model.data_preprocessor.std,
                inplace=True,
            )(data_batch_prepro["inputs"][0])
            model.training = False  # avoid missing arguments error in some models
            self._evaluator_process(evaluators[step + 1], data_batch_prepro, model)

        return data_batch_prepro
