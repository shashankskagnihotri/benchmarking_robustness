from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch import Generator
from dataloader import get_dataset
from typing import Literal
from torch.utils.data import random_split, Dataset
from torch import Generator


# TODO: fix MPISintel corruption and other split loading


def get_dataset(
    dataset_name: str,
    data_path: str,
    architecture_name: str,
    split: Literal["train", "validation", "test", "corrupted"],
    debug: bool = False,
    random_seed: int = 42,
):
    """
    Load and return the dataset object based on the provided dataset name.

    Args:
        random_seed:
        debug:
        dataset_name (str): The name of the dataset to load. Supported options are
                            'sceneflow', 'sintel', 'kitti', 'kitti2015', 'eth3d',
                            and 'mpisintel'.
        data_path (str): The directory where the dataset is located.
        split (str): The data split to load. Common values include 'train', 'test', etc.
        architecture_name (str): The name of the architecture for which the dataset
                                will be used.

    Returns:
        An instance of the dataset class corresponding to the dataset_name provided.
    """

    dataset_name = dataset_name.lower()
    dataset: Dataset

    print(f"Loading {dataset_name} dataset")
    if dataset_name == "sceneflow":
        from .sceneflow import SceneFlowFlyingThings3DDataset

        if split == "validation":
            dataset = SceneFlowFlyingThings3DDataset(
                data_path, architecture_name, split="train"
            )
            _, dataset = perform_train_test_split(dataset, 0.8, random_seed)
        else:
            # Note: train / test / corruption split inside SceneFlowFlyingThings3DDataset
            dataset = SceneFlowFlyingThings3DDataset(data_path, architecture_name, split)
    elif dataset_name == "sintel":
        from .mpisintel import MPISintelDataset

        return MPISintelDataset(data_path, architecture_name, split)

    elif dataset_name == "kitti" or dataset_name == "kitti2015":
        from .kitti2015 import KITTIBaseDataset

        if split == "test":
            dataset = KITTIBaseDataset(data_path, architecture_name, split="train")
            _, dataset = perform_train_test_split(dataset, 0.85, random_seed)
        else:
            # Note: train / val split inside KITTIBaseDataset
            dataset = KITTIBaseDataset(data_path, architecture_name, split)

    elif dataset_name == "eth3d":
        from .eth3d import ETH3DDataset

        return ETH3DDataset(data_path, architecture_name, split)
    elif dataset_name == "mpisintel":
        from .mpisintel import MPISintelDataset

        return MPISintelDataset(data_path, architecture_name, split)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Return less entries if in debug mode
    if debug:
        dataset = Subset(dataset, list(range(10)))

    return dataset


### START - Get data loaders for CFNet and GWCNet
def get_default_loader(args, architecture_name: str, debug: bool, random_seed: int):
    train_img_loader = DataLoader(
        get_dataset(
            args.dataset, args.datapath, architecture_name, "train", debug, random_seed
        ),
        args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )
    val_img_loader = DataLoader(
        get_dataset(
            args.dataset,
            args.datapath,
            architecture_name,
            "validation",
            debug,
            random_seed,
        ),
        args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )
    test_img_loader = DataLoader(
        get_dataset(
            args.dataset, args.datapath, architecture_name, "test", debug, random_seed
        ),
        args.test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    return test_img_loader, train_img_loader, val_img_loader


def perform_train_test_split(
    dataset: Dataset, fraction: float, random_seed: int
) -> tuple[Dataset, Dataset]:
    subset1_size = int(fraction * len(dataset))  # fraction for the first subset
    subset2_size = len(dataset) - subset1_size  # remainder for the second subset

    generator = Generator()
    generator.manual_seed(seed=random_seed)
    subset1, subset2 = random_split(dataset, [subset1_size, subset2_size], generator)
    return subset1, subset2


# def perform_train_test_val_split(dataset: Dataset, random_seed: int) -> tuple[Dataset, Dataset, Dataset]:
#     val_size = int(0.2 * len(dataset))  # 20% for validation
#     test_size = int(0.1 * len(dataset))  # 10% for testing
#     train_size = len(dataset) - val_size - test_size
#
#     generator = Generator()
#     generator.manual_seed(seed=random_seed)
#     train_subset, val_subset, test_dataset = random_split(
#         dataset,
#         [train_size, val_size, test_size],
#         generator
#     )
#     return test_dataset, train_subset, val_subset
