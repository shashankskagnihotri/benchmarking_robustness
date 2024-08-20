import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from mmengine import Config


def process_row(row):
    arch, backbone, boxAP, config, weights, dataset = row
    print(arch, backbone, boxAP, config, weights, dataset)

    # Read and save the config file to get resolved config
    cfg = Config.fromfile(os.path.join("mmdetection", config))
    save_path = os.path.join("models", f"{arch}_{backbone}")
    os.makedirs(save_path, exist_ok=True)
    cfg.dump(os.path.join(save_path, f"{arch}_{backbone}.py"))

    # Download the weights file
    if os.path.exists(os.path.join(save_path, "latest.pth")):
        print("Weights file already exists, skipping download")
    else:
        urllib.request.urlretrieve(weights, os.path.join(save_path, "latest.pth"))


def main():
    df = pd.read_csv("ready_made_models.csv")

    # Strip whitespace from all string values in the DataFrame
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    with ThreadPoolExecutor() as executor:
        # Submit each row to the thread pool executor
        executor.map(process_row, df.itertuples(index=False, name=None))


if __name__ == "__main__":
    main()
