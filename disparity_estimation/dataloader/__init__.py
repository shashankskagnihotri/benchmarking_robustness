def get_dataset(dataset_name:str, datadir:str, split:str, architeture_name:str):

    dataset_name = dataset_name.lower()

    print(f'Loading {dataset_name} dataset')
    if dataset_name == 'sceneflow':
        from .sceneflow import SceneFlowFlyingThings3DDataset
        isTrain = True if split == 'train' else False
        return SceneFlowFlyingThings3DDataset(datadir, architeture_name, isTrain)
    
    elif dataset_name == 'sintel':
        from .mpisintel import MPISintelDataset
        return MPISintelDataset(datadir, architeture_name, split)
    
    elif dataset_name == 'kitti2015':
        from .kitti2015 import KITTIBaseDataset
        return KITTIBaseDataset(datadir, architeture_name, split)
    # elif dataset_name == 'eth3d':
    #     # from eth3d     import ETH3DDataloader
        # return ETH3DDataloader(datadir, split)
    # elif dataset_name == 'mpisintel':
    #     return MPISintelDataloader(datadir, split)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')