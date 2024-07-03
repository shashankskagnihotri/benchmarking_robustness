import argparse
import os
import sys
import mlflow

mlflow.set_tracking_uri("/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/mlflow/")

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', required=True, help='Specify an architecture')
parser.add_argument('-m', '--mode', required=True, help='Specify whether to train or test the model', choices=['train', 'test'])

args, args_unknown = parser.parse_known_args()
# name = args.name
# print(name)

# Add the CFNet directory to the PYTHONPATH
cfnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CFNet'))
sys.path.append(cfnet_path)

# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
with mlflow.start_run(experiment_id='128987742873377588'):

    mlflow.log_params(vars(args))
    if args.architecture.lower() == "cfnet":
        
        if args.mode.lower() == "train":
            from CFNet import main
            main.train()
        else:
            from CFNet import test
            main.test()