from .sceneflow.sceneflow import SceneFlowFlyingThings3DDataset
from .kitti2015 import Kitti2015Dataloader
from .Kitti2015.kitti2015 import KITTIBaseDataset
# from .eth3d     import ETH3DDataloader
from .mpisintel import MPISintelDataloader

def get_dataset(dataset_name:str, datadir:str, split:str, architeture_name:str):
    print(f'Loading {dataset_name} dataset')
    if dataset_name.lower() == 'sceneflow':
        isTrain = True if split == 'train' else False
        return SceneFlowFlyingThings3DDataset(datadir, architeture_name, isTrain)
    elif dataset_name == 'sintel':
        return MPISintelDataset(datadir, architeture_name)
    # elif dataset_name == 'kitti2015':
    #     return Kitti2015Dataloader(datadir, split)
    # elif dataset_name == 'eth3d':
    #     return ETH3DDataloader(datadir, split)
    # elif dataset_name == 'mpisintel':
    #     return MPISintelDataloader(datadir, split)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')