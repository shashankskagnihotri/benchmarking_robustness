from .sceneflow.sceneflow import SceneFlowFlyingThings3DDataset
from .kitti2015.kitti2015 import Kitti2015Dataset
from .eth3d.eth3d         import ETH3DDataset
from .mpisintel.mpisintel import MPISintelDataset

def get_dataset(datadir : str , dataset_name : str, split : str, architeture_name: str):
    if dataset_name == 'sceneflow':
        isTrain = True if split == 'train' else False
        return SceneFlowFlyingThings3DDataset(datadir, isTrain, architeture_name)
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