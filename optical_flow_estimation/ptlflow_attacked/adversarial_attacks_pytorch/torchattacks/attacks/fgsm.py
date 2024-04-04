import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.squeeze(0)
        labels = labels.squeeze(0)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        inputs = {"images": images.unsqueeze(0)}
        outputs = self.get_logits(inputs)
        outputs_tensor = outputs["flows"].squeeze(0)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs_tensor, target_labels)
        else:
            cost = loss(outputs_tensor, labels)

        # Update adversarial images
        grads = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        image_1_grad = grads[0].unsqueeze(0)
        image_2_grad = grads[1].unsqueeze(0)
        

        image_1 = images.squeeze(0)[0].unsqueeze(0).detach().to(self.device)
        image_2 = images.squeeze(0)[1].unsqueeze(0).detach().to(self.device)
        
        image_adv_1 = image_1 + self.eps * image_1_grad.sign()
        image_adv_2 = image_2 + self.eps * image_2_grad.sign()
        image_adv_1 = torch.clamp(image_adv_1, min=0, max=1).detach()
        image_adv_2 = torch.clamp(image_adv_2, min=0, max=1).detach()

        images_adv = torch.torch.cat((image_adv_1, image_adv_2)).unsqueeze(0)
        inputs_adv = {"images": images_adv}

        return inputs_adv
