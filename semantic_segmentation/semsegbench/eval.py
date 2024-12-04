from semsegbench import bench_df
import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
from mmseg.apis import MMSegInferencer

def load_model(model_name: str, backbone: str = None, dataset: str = None, crop_size: str = None,
               load_without_weights = False):
    """
    Load and return model specified by parameters
    - model_name (str): either just architecture name like 'segformer' or whole config filename like 'segformer_mit-b0_8xb2-160k_ade20k-512x512'
    - (optional) backbone (str): for example 'mit-b0'
    - (optional) dataset (str): for example 'ade20k'
    - (optional) crop_size (str): for example '512x512'

    - load_without_weights (bool) default False: if True, no checkoint file will be searched and model will be loaded withou weights
    """
    model_architecture = model_name.split("_")[0]
    architecture_config_path = Path(os.path.join(current_dir, f"../mmsegmentation/configs/{model_architecture}"))
    if not architecture_config_path.is_dir():
        raise FileNotFoundError(f"No config for model with architecture name '{model_architecture}' existent!")
    
    # if model_name contains underscore --> expect whole config filename
    if len(model_name.split("_")) > 1:
        config_path = architecture_config_path / Path(model_name+".py")
        if not config_path.is_file():
            raise FileNotFoundError(f"No config found for model {model_name}")
    else:
        configs_in_dir = [config_path for config_path in architecture_config_path.iterdir() if config_path.stem.startswith(model_architecture)]
        
        # filter by backbone, dataset and crop_size if given
        if backbone:
            configs_in_dir = [config_path for config_path in configs_in_dir if backbone in config_path.stem]
        if dataset:
            configs_in_dir = [config_path for config_path in configs_in_dir if dataset in config_path.stem]
        if crop_size:
            configs_in_dir = [config_path for config_path in configs_in_dir if crop_size in config_path.stem]

        if len(configs_in_dir) > 1:
            error_message = "More than one model found with given parameters {'model_name': " + model_name
            if backbone:
                error_message += ", 'backbone': " + backbone
            if dataset:
                error_message += ", 'dataset': " + dataset
            if crop_size:
                error_message += ", 'crop_size': " + crop_size
            error_message += "}\n"
            error_message += "Following configs found: " + str([config_path.stem for config_path in configs_in_dir]) + "\n\n"
            error_message += "If a value for each of the parameters 'model_name', 'backbone', 'dataset' and 'crop_size' were given, please enter the exact config_name in the paramter 'model_name'."
            
            raise ValueError(error_message)
        else:
            config_path = configs_in_dir[0]
    
    if load_without_weights:
        checkpoint_path = None
    else:
        # search for checkpoint file
        checkpoint_files_path = Path(os.path.join(current_dir, f"../checkpoint_files/{model_architecture}"))
        if not checkpoint_files_path.is_dir():
            raise FileNotFoundError(f"No checkpoint file found for config '{config_path.stem}'\n\n checkpoint file names have to start like the corresponding config file name")

        # get pth files that start with same filename as config filename
        matching_checkpoint_files = [checkpoint_file_path for checkpoint_file_path in checkpoint_files_path.iterdir() if checkpoint_file_path.stem.startswith(config_path.stem)]
        if len(matching_checkpoint_files) == 0:
            raise ValueError(f"No checkpoint file found for config '{config_path.stem}'\n\n checkpoint file names have to start like the corresponding config file name")
        elif len(matching_checkpoint_files) > 1:
            raise ValueError(f"More than one checkpoint file found for config '{config_path.stem}'")
        else:
            checkpoint_path = matching_checkpoint_files[0]
            print("found checkpoint_path")
    
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)
    model_str_path = str(config_path)
    
    import pdb
    pdb.set_trace()
    inferencer = MMSegInferencer(model=model_str_path, weights=checkpoint_path)
    model = inferencer.model
    return model



if __name__ == "__main__":
    load_model("segformer", "mit-b0", "cityscapes", "1024x1024")