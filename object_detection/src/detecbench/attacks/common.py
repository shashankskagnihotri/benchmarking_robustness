import torch


def preprocess_data(data_batch: dict, runner):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)
    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    mean = runner.model.data_preprocessor.mean
    std = runner.model.data_preprocessor.std
    images = denorm(images, mean, std)
    return data_batch_prepro, images

# restores the tensors to their original scale
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def denorm(batch, mean, std):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to("cuda")
    if isinstance(std, list):
        std = torch.tensor(std).to("cuda")

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def evaluator_process(evaluator, data_batch_prepro, model):
    model.training = False  # avoid missing arguments error in some models
    model.bbox_head.adv_attack = False # needed for handling issue in reppoint heads

    with torch.no_grad():
        outputs = model(**data_batch_prepro, mode="predict")

    model.bbox_head.adv_attack = True
    evaluator.process(data_samples=outputs, data_batch=data_batch_prepro)
