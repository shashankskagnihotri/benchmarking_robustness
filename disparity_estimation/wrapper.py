import argparse
import os
import sys
import mlflow

# Add the CFNet path
# cfnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CFNet'))
# sys.path.append(cfnet_path)
# cfnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Gw'))
# sys.path.append(cfnet_path)

mlflow.set_tracking_uri("/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/mlflow/")
parser = argparse.ArgumentParser()
parser.add_argument('--architecture', required=True, help='Specify an architecture') #, choices=['cfnet', 'gwcnet', 'psmnet', 'sttr', 'sttr-light']
parser.add_argument('--scenario', required=True, help='Specify whether to train or test the architecture', choices=['train', 'test', 'attack', 'commoncorruption'])
parser.add_argument('--commoncorruption', required=False, help='Specify the name of the common corruptions to apply. --phase must be test')
parser.add_argument('--severity', required=False, help='Specify the severity level of the common corruptions to apply. --phase must be test and --commoncorruption must be specified')
parser.add_argument('--attack_type', required=False, help='Specify the attack to apply. --phase must be test')

args, unknown = parser.parse_known_args()
args.scenario = args.scenario.lower()
args.architecture = args.architecture.lower()


if args.scenario == "commoncorruption" and (args.commoncorruption is None or args.severity is None):
    raise ValueError("If --scenario is commoncorruption, --commoncorruption and --severity must be specified")

if args.scenario == "attack" and args.attack_type is None:
    raise ValueError("If --scenario is attack, --attack must be specified")

# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
with mlflow.start_run(experiment_id='128987742873377588'):
    mlflow.log_params(vars(args))

    if args.architecture == "cfnet":
        from CFNet import main
        print("Loaded cfnet")
    elif args.architecture == "gwcnet":
        from GwcNet import main
        print("Loaded gwcnet")
    else:
        raise ValueError("Architecture (/ Architecture depricated) not recognized")

    print(f"Running {args.scenario} scenario for {args.architecture}")
        
    if args.scenario == "train":
        mlflow.set_tag("mlflow.runName", f"{args.architecture}_train")
        main.train()

    elif args.scenario == "test":
        mlflow.set_tag("mlflow.runName", f"{args.architecture}_test")
        main.test()

    elif args.scenario == "attack":
        mlflow.set_tag("mlflow.runName", f"{args.architecture}_attack_{args.attack_type}")
        main.attack(attack_type=args.attack_type)
    elif args.scenario == "commoncorruption":

        if args.commoncorruption not in args.unknown:
            raise ValueError("Common corruption not recognized")

        mlflow.set_tag("mlflow.runName", f"{args.scenario}_commoncorruption_{args.commoncorruption}_{args.severity}")
        main.test()
    else:
        raise ValueError("Scenario not recognized")
