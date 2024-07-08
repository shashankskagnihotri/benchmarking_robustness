import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, x1, x2, labels):
        x1 = x1.clone().detach().to(self.device)
        x2 = x2.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_x1 = x1.clone().detach()
        adv_x2 = x2.clone().detach()

        if self.random_start:
            # Start at a uniformly random point
            adv_x1 = adv_x1 + torch.empty_like(adv_x1).uniform_(-self.eps, self.eps)
            adv_x2 = adv_x2 + torch.empty_like(adv_x2).uniform_(-self.eps, self.eps)
            adv_x1 = torch.clamp(adv_x1, min=0, max=1).detach()
            adv_x2 = torch.clamp(adv_x2, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_x1.requires_grad = True
            adv_x2.requires_grad = True
            outputs = self.model(adv_x1, adv_x2)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad_x1, grad_x2 = torch.autograd.grad(cost, [adv_x1, adv_x2], 
                                                   retain_graph=False, create_graph=False)

            adv_x1 = adv_x1.detach() + self.alpha * grad_x1.sign()
            adv_x2 = adv_x2.detach() + self.alpha * grad_x2.sign()

            delta_x1 = torch.clamp(adv_x1 - x1, min=-self.eps, max=self.eps)
            delta_x2 = torch.clamp(adv_x2 - x2, min=-self.eps, max=self.eps)

            adv_x1 = torch.clamp(x1 + delta_x1, min=0, max=1).detach()
            adv_x2 = torch.clamp(x2 + delta_x2, min=0, max=1).detach()

        return adv_x1, adv_x2