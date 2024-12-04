import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

excel_path = os.path.join(current_dir, "Config-Paths.xlsx")

# load benchmark table
bench_df = pd.read_excel(excel_path, sheet_name = "results", index_col = 0, header = None)

# preprocess table (change dtypes and orientation)
bench_df = bench_df.T

numeric_columns = [col for col in bench_df.columns if col.startswith(("aAcc", "mIoU", "mAcc"))] + ["num_params",]
bench_df[numeric_columns] = bench_df[numeric_columns].apply(pd.to_numeric, errors = "coerce")
bench_df["time_proposed"] = pd.to_datetime(bench_df["time_proposed"])

__all__ = ["bench_df",]