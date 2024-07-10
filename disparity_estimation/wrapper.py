import argparse
import os
import sys
import mlflow

mlflow.set_tracking_uri("/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/mlflow/")
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', required=True, help='Specify an architecture')
parser.add_argument('-p', '--phase', required=False, help='Specify whether to train or test the model', choices=['train', 'test'])
parser.add_argument('-c', '--commoncorruption', required=False, help='Specify the name of the common corruptions to apply')
parser.add_argument('-s', '--severity', required=False, help='Specify the severity level of the common corruptions to apply')

args, unknown = parser.parse_known_args()

if args.commoncorruption is not None:
    pass
# name = args.name
# print(name)

# Add the CFN#
cfnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CFNet'))
sys.path.append(cfnet_path)

# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
with mlflow.start_run(experiment_id='128987742873377588'):

    if args.architecture.lower() == "cfnet":
        from CFNet import main
        if "" == "train":
            main.train()
        else:
            run_name = f"CFNet_corruption_brightness_severity_5"
            main.test()