from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
plt.ioff()


def get_best_mIoU_iter(work_dir_path_to_model: Path):

    val_scalars_list = list()
    train_scalars_list = list()
    
    for run_folder in work_dir_path_to_model.iterdir():
        
        if run_folder.is_dir():
            
            scalars_file = run_folder / Path("vis_data/scalars.json")

            if scalars_file.exists():
            
                with open(scalars_file) as fp:
                    
                    for line in fp:

                        if "mIoU" in line:
                            val_scalars_list.append(json.loads(line.rstrip()))
                        else:
                            train_scalars_list.append(json.loads(line.rstrip()))
                    
    result = None

    if len(train_scalars_list) > 0:
        train_scalars = pd.DataFrame(train_scalars_list)
        train_scalars.to_csv(work_dir_path_to_model / "train_scalars.csv", index = False)

    if len(val_scalars_list) > 0:
        val_scalars = pd.DataFrame(val_scalars_list)
        val_scalars.to_csv(work_dir_path_to_model / "val_scalars.csv", index = False)
        
        mIoU_max_i = val_scalars.mIoU.idxmax()

        result = (val_scalars.loc[mIoU_max_i, "step"], val_scalars.loc[mIoU_max_i, "mIoU"])

    else:
        warnings.warn("<function> get_best_mIoU_iter: emtpy list `val_scalars_list`")
    
    if len(train_scalars_list) > 0 and len(val_scalars_list) > 0:
        ax = train_scalars.set_index("step")["loss"].plot(lw = 0.1)
        ax.set_ylabel("loss")
        val_scalars.set_index("step")["mIoU"].plot(ax = ax, secondary_y = True, style = ".")
        plt.legend()
        fig = ax.get_figure()
        fig.savefig(work_dir_path_to_model / Path("training_progress.png"))
        plt.close()
    
    return result



if __name__ == "__main__":
    work_dir = Path("work_dirs")
    for work_dir_path_to_model in work_dir.iterdir():
        result = get_best_mIoU_iter(work_dir_path_to_model)
        print(f"{work_dir_path_to_model.stem}: ", result)