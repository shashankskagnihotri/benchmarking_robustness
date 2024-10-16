from pathlib import Path
import pandas as pd 
import numpy as np
import json
from tqdm import tqdm

def create_cor_csv(model_path:Path):
    metrics_list = ["aAcc","mIoU","mAcc"]
    attacks_list = ["cospgd", "segpgd", "pgd"]
    norms_list = ["linf", "l2"]

    series_index = ['_'.join([metric, attack, norm]) for attack in attacks_list for norm in norms_list for metric in metrics_list]

    results = pd.Series(series_index).str.split("_", expand = True)
    results.index = pd.Index(series_index)
    results.columns = ["metric", "attack", "norm"]
    results["epsilon"] = 8
    results.loc[results.norm == "l2", "epsilon"] = 64
    results["alpha"] = 0.01
    results.loc[results.norm == "l2", "alpha"] = 0.1
    results["iterations"] = 20

    results["value"] = np.nan

    for index, test_row in results.iterrows():
        
        test_path = model_path / Path(f"attack_{test_row.attack}/norm_{test_row.norm}/iterations_{test_row.iterations}/epsilon_{test_row.epsilon}/alpha_{test_row.alpha}")
        
        if not test_path.exists():
            print(f"nonexistent path for {test_path}")
            continue

        run_directories = []
        for dir in test_path.iterdir():
            # only keep dir if is directory
            if not dir.is_dir():
                continue
            
            # only keep dir if is a run directory (starts with date -> can be parsed to int)
            try:
                int(dir.stem.split("_")[0])
            except ValueError:
                continue

            run_directories.append(dir)
        
        run_path = max(run_directories, key = lambda x: x.name)

        try:
            with open(run_path/Path(run_path.stem + ".json")) as fp:
                run_dict = json.load(fp)
        except FileNotFoundError as e:
            print(e)
            continue

        results.loc[index, "value"] = run_dict[test_row.metric]
    
    results["value"].to_csv(model_path/Path(f"attacks_results_{model_path.stem}.csv"), header=False, index=False)

if __name__ == "__main__":
    workdir = Path("./")
    for dataset in ["cityscapes",]:
        print()
        print(dataset)
        dataset_path = workdir/Path(dataset+"/")
        for architecture_path in dataset_path.iterdir():
            for model_path in tqdm(architecture_path.iterdir(),desc=f"{architecture_path.stem}",total=len(list(architecture_path.iterdir()))):
                if model_path.is_file():
                    continue
                create_cor_csv(model_path)

