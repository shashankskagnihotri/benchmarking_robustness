from sceneflow import SceneFlowFlyingThings3DDataset
from kitti2015 import KITTIBaseDataset
# from eth3d     import ETH3DDataloader
from mpisintel import MPISintelDataset

def get_dataset(dataset_name:str, datadir:str, split:str, architeture_name:str):

    dataset_name = dataset_name.lower()

    print(f'Loading {dataset_name} dataset')
    if dataset_name == 'sceneflow':
        isTrain = True if split == 'train' else False
        return SceneFlowFlyingThings3DDataset(datadir, architeture_name, isTrain)
    
    elif dataset_name == 'sintel':
        return MPISintelDataset(datadir, architeture_name, split)
    elif dataset_name == 'kitti2015':
        return KITTIBaseDataset(datadir, architeture_name, split)
    # elif dataset_name == 'eth3d':
    #     return ETH3DDataloader(datadir, split)
    # elif dataset_name == 'mpisintel':
    #     return MPISintelDataloader(datadir, split)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')