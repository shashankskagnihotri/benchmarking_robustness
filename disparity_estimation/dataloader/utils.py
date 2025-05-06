def get_checkpoint_path(dataset:str, model:str) -> str:
    checkpoints = {
        "sceneflow": {
            "gwcnet-gc": "pretrained_weights/gwcnet/sceneflow/gwcnet-gc/checkpoint_000015.ckpt",
            "gwcnet-g": "pretrained_weights/gwcnet/sceneflow/gwcnet-g/checkpoint_000015.ckpt",
            "cfnet": "pretrained_weights/cfnet/sceneflow_pretraining.ckpt",
            "sttr": "pretrained_weights/sttr/sceneflow_pretrained_model.pth.tar",
            "sttr-light": "pretrained_weights/sttr-light/sttr_light_sceneflow_pretrained_model.pth.tar",
        },
        "kitti2015": {
            "gwcnet-g": "pretrained_weights/gwcnet/kitti15/gwcnet-g/best.ckpt",
            "sttr": "pretrained_weights/sttr/kitti_finetuned_model.pth.tar",
        }
    }
    
    if dataset not in checkpoints:
        raise ValueError(f"Checkpoint loading: Dataset {dataset} not recognized")
    
    if model not in checkpoints[dataset]:
        raise ValueError(f"Checkpoint loading: Model {model} not recognized for dataset {dataset}")
    
    return checkpoints[dataset][model]


def get_dataset_path(dataset:str, corruption_type:str='no_corruption', severity_level:int=0) -> str:
    dataset_paths = {
        "sceneflow": f"datasets/FlyingThings3D/Common_corruptions/{corruption_type}/severity_{severity_level}",
        "kitti2015": f"datasets/KITTI_2015/Common_corruptions/{corruption_type}/severity_{severity_level}",
    }

    if dataset not in dataset_paths:
        raise ValueError(f"Dataset {dataset} not recognized")
    
    return dataset_paths[dataset]