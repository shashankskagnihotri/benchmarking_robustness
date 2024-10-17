import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

plt.ioff()

def visualize_training(fpath):
    val_scalars_list = list()
    train_scalars_list = list()
    scalars_file = fpath / Path("vis_data/scalars.json")
    with open(scalars_file) as fp:
        for line in fp:

            if "mIoU" in line:
                val_scalars_list.append(json.loads(line.rstrip()))
            else:
                train_scalars_list.append(json.loads(line.rstrip()))

    if len(train_scalars_list) > 0:
        train_scalars = pd.DataFrame(train_scalars_list)
    if len(val_scalars_list) > 0:
        val_scalars = pd.DataFrame(val_scalars_list)

    if len(train_scalars_list) > 0 and len(val_scalars_list) > 0:
        ax = train_scalars.set_index("step")["loss"].plot(lw = 0.1)
        ax.set_ylabel("loss")
        val_scalars.set_index("step")["mIoU"].plot(ax = ax, secondary_y = True, style = ".")
        plt.legend()
        fig = ax.get_figure()
        fig.savefig(fpath/ Path("training_visualized.png"))
        plt.close()

if __name__ == "__main__": 
    fpath = Path("./work_dirs/unet-s5-d16_fcn_4x4-160k_pascal_voc-512x512_stride-340x340_lr01/20240718_184328")
    visualize_training(fpath)