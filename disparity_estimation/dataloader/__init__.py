def get_dataset(dataset:str, dataset_path:str, split:str, architeture_name:str):

    dataset = dataset.lower()
    
    print(f'Loading {dataset} dataset')
    if dataset == 'sceneflow':
        from .sceneflow import SceneFlowFlyingThings3DDataset
        print("Dataset path: ", dataset_path)
        return SceneFlowFlyingThings3DDataset(dataset_path, architeture_name, split)
    
    elif dataset == 'sintel':
        from .mpisintel import MPISintelDataset
        return MPISintelDataset(dataset_path, architeture_name, split)
    
    elif dataset == 'kitti' or dataset == 'kitti2015':
        from .kitti2015 import KITTIBaseDataset
        if split == 'test':
            return None
        return KITTIBaseDataset(dataset_path, architeture_name, split)
    elif dataset == 'eth3d':
        from .eth3d     import ETH3DDataset
        #isTrain = True if split == 'train' else False
        return ETH3DDataset(dataset_path, architeture_name, split)
    elif dataset == 'mpisintel':
        from .mpisintel import MPISintelDataset
        return MPISintelDataset(dataset_path, architeture_name, split)
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')


### START - Get data loaders for CFNet and GWCNet

import random
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from dataloader import get_dataset



def get_data_loader_1(args, architeture_name):
    train_dataset = get_dataset(
        args.dataset, args.dataset_path, architeture_name=architeture_name, split="train"
    )

    test_dataset = get_dataset(
        args.dataset, args.dataset_path, architeture_name=architeture_name, split="test"
    )

    # TODO: Change for inferance, add if that checks if only inference is performed, then only test data is loaded 
    if "kitti" in args.dataset.lower() or "mpisintel" in args.dataset.lower():  # Define split sizes
        val_size = int(0.2 * len(train_dataset))  # 20% for validation
        test_size = int(0.1 * len(train_dataset))  # 10% for testing
        train_size = len(train_dataset) - val_size - test_size

        # Split the dataset, because kitti has no test split
        train_subset, val_subset, test_dataset = random_split(
            train_dataset, [train_size, val_size, test_size]
        )
    else:
        val_size = int(0.2 * len(train_dataset))  # 20% for validation
        train_size = len(train_dataset) - val_size

        # Split the dataset
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    del train_dataset


    fast_dev_run = False
    if fast_dev_run == True:
        # Create small subsets for fast_dev_run
        fast_dev_run_size = 10  # Number of data points to use in fast_dev_run

        # Create subsets for training, validation, and testing
        train_indices = list(range(fast_dev_run_size))
        val_indices = list(range(fast_dev_run_size, 2 * fast_dev_run_size))
        test_indices = list(range(fast_dev_run_size))

        train_subset = Subset(train_subset, train_indices)
        val_subset = Subset(val_subset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)

    ValImgLoader = DataLoader(
        val_subset, args.batch_size, shuffle=False, num_workers=8, drop_last=True
    )
    TrainImgLoader = DataLoader(
        train_subset, args.batch_size, shuffle=False, num_workers=8, drop_last=True
    )
    TestImgLoader = DataLoader(
        test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False
    )

    return TrainImgLoader, ValImgLoader, TestImgLoader