from typing import Optional, Union, List, Literal
from pathlib import Path
from os import PathLike

import pdb

PathType = Union[str, Path, PathLike]

import pandas as pd

class SemSegWrapper:
    def __init__(self, df: Optional[pd.DataFrame] = None, load_excel: bool = False,
                 excel_path: Optional[PathType] = None, excel_sheet: Optional[str] = None):
        self.df = df
        self.load_excel = load_excel
        self.excel_path = excel_path
        self.excel_sheet = excel_sheet
        
        if self.load_excel:
            
            if self.excel_path is None:
                raise ValueError("If 'load_excel=True', then a path has to be given in the parameter 'excel_path'!")
            else:
                df = pd.read_excel(excel_path, sheet_name = self.excel_sheet, index_col = 0, header = None)
                df = df.T

                # change data types of columns
                numeric_columns = [col for col in df.columns if col.startswith(("aAcc", "mIoU", "mAcc"))] + ["num_params",]
                df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors = "coerce")
                df["time_proposed"] = pd.to_datetime(df["time_proposed"])

                self.df = df
    
    def get_result(self,
                   filters: Optional[dict] = None,
                   return_metric: Optional[Union[Literal["aAcc", "mIoU", "mAcc"], List[Literal["aAcc", "mIoU", "mAcc"]]]] = None,
                   test_type: Optional[Union[Literal["clean", "2d", "cospgd", "segpgd", "pgd", "acdc"], List[Literal["clean", "2d", "cospgd", "segpgd", "pgd", "acdc"]]]] = None,
                   attack_cfg: Optional[dict] = None,
                   corruption_cfg: Optional[dict] = None):
        
        result_df = self.df.copy()
        original_columns = result_df.columns
        meta_columns = ["architecture", "backbone", "type_backbone", "dataset", "crop_size"]

        if filters is not None:
            for filter_key, filter_value in filters.items():
                result_df = result_df[result_df[filter_key] == filter_value]
        
        if return_metric is not None:
            if isinstance(return_metric, str):
                return_metric = [return_metric, ]

            metric_columns = [col for col in result_df.columns if col.startswith(tuple(return_metric))]
            result_df = result_df[metric_columns]
        
        if test_type is not None:
            if isinstance(test_type, str):
                test_type = [test_type, ]
            test_type_columns = [col for col in result_df.columns if any(type in col.split("_") for type in test_type)]
            result_df = result_df[test_type_columns]
        
        if len(result_df) > 1:
            result_df = self.df.copy().loc[result_df.index, meta_columns+list(result_df.columns)]
        
        if result_df.empty:
            return None
        else:
            return result_df
        

if __name__ == "__main__":
    wrapper = SemSegWrapper(load_excel = True, excel_path = "Config-Paths.xlsx", excel_sheet = "results")
    pdb.set_trace()
    print("End")