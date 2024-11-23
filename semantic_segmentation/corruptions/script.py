# import os
# # Get the directory of the current script
# script_directory = os.path.dirname(os.path.abspath(__file__))
# # Construct the path to the 'work_dirs' directory
# work_dirs_path = os.path.join(script_directory, 'work_dirs')
# import pdb 
# pdb.set_trace()

from pathlib import Path
import pandas as pd 
import json
from tqdm import tqdm

def create_cor_csv(model_path:Path):
    metrics_list = ["aAcc","mIoU","mAcc"]
    corruptions_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    series_index = [metric + "_2d_" + corruption for corruption in corruptions_list for metric in metrics_list ]

    results = pd.Series(index=series_index, dtype=float)

    for corruption_path in model_path.iterdir():
        if corruption_path.is_file():
            continue
        run_folder = max((f for f in corruption_path.iterdir() if f.is_dir()), key=lambda x: x.name)
        
        try:
            with open(run_folder/Path(run_folder.stem + ".json")) as fp:
                run_dict = json.load(fp)
        except FileNotFoundError as e:
            print(e)
            continue
            
        results.loc[f"aAcc_2d_{corruption_path.stem}"] = run_dict["aAcc"]
        results.loc[f"mIoU_2d_{corruption_path.stem}"] = run_dict["mIoU"]
        results.loc[f"mAcc_2d_{corruption_path.stem}"] = run_dict["mAcc"]

    results.to_csv(model_path/Path(f"corruption_result_{model_path.stem}.csv"), header=False, index=False, decimal=",", sep=";")

if __name__ == "__main__":
    workdir = Path("work_dirs/")
    for dataset in ["ade20k","cityscapes","pascalvoc"]:
        print()
        print(dataset)
        dataset_path = workdir/Path(dataset+"/")
        for architecture_path in dataset_path.iterdir():
            for model_path in tqdm(architecture_path.iterdir(),desc=f"{architecture_path.stem}",total=len(list(architecture_path.iterdir()))):
                if model_path.is_file():
                    continue
                create_cor_csv(model_path)

