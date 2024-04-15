import os
import pandas as pd
from tabulate import tabulate


def collect_results(main_folder="work_dirs"):
    df = pd.DataFrame()
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        for json_file in os.listdir(subfolder_path):
            json_path = os.path.join(
                subfolder_path, json_file
            )  # might not a json at this point

            if json_file == "args.json":
                df_args = pd.read_json(json_path, typ="series").to_frame(subfolder).T
            elif json_file == "cfg.json":
                pass
            elif json_file == "metrics.json":
                df_metrics = pd.read_json(json_path, typ="series").to_frame(subfolder).T
            else:
                continue

        df_subfolder = df_args.join(df_metrics)
        df = pd.concat([df, df_subfolder])

    print(tabulate(df, headers=df.columns))

    output_file = os.path.join(main_folder, "attack_results.csv")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    collect_results("slurm/logs/")
