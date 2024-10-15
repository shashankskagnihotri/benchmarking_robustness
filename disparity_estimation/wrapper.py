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
parser.add_argument('--model', required=True, help='Specify an model', choices=['cfnet', 'gwcnet', 'gwcnet-g', 'gwcnet-gc', 'psmnet', 'sttr', 'sttr-light'])
parser.add_argument('--scenario', required=True, help='Specify whether to train or test the model', choices=['train', 'test', 'attack', 'commoncorruption'])
parser.add_argument('--dataset', required=True, help='Specify the dataset to use', choices=['sceneflow', 'mpisintel', 'kitti', 'kitti2015', 'eth3d', 'mpisintel'])
parser.add_argument('--commoncorruption', required=False, help='Specify the name of the common corruptions to apply. --phase must be test')
parser.add_argument('--severity', required=False, help='Specify the severity level of the common corruptions to apply. --phase must be test and --commoncorruption must be specified')
parser.add_argument('--attack_type', required=False, help='Specify the attack to apply. --phase must be test')
parser.add_argument('--experiment', required=False, default='debug', type=str, choices=['debug', 'Common_Corruptions'], help='Specify the experiment to log to')

args, unknown = parser.parse_known_args()
args.scenario = args.scenario.lower()
args.model = args.model.lower()

if args.scenario == "commoncorruption" and (args.commoncorruption is None or args.severity is None):
    raise ValueError("If --scenario is commoncorruption, --commoncorruption and --severity must be specified")

if args.scenario == "attack" and args.attack_type is None:
    raise ValueError("If --scenario is attack, --attack must be specified")

# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run

# Get experiment_id dynamically from args.experiment
experiment_name = f"{args.experiment}_{args.model}_{args.dataset}"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    # print(f"Experiment '{experiment_name}' does not exist. Creating a new experiment.")
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
    # raise ValueError(f"Experiment '{experiment_name}' does not exist")
else:
    print(f"Experiment '{experiment_name}' exists.")
    experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")
    print(f"Experiment state: {experiment.lifecycle_stage}")

    


# Rest of the code...

with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_params(vars(args))

    if args.model == "cfnet":
        from CFNet import main
        print("Loaded cfnet")
    elif args.model in "gwcnet-gc":
        from GwcNet import main
        print("Loaded gwcnet")
    elif args.model == "sttr":
        # import importlib  
        # from importlib.import_module("foo-bar") import main
        from sttr import main # type: ignore
        print("Loaded sttr")
    elif args.model == "sttr-light":
        from sttr_light import main
        print("Loaded sttr-light")
        
    else:
        raise ValueError("Architecture/Model (/ Architecture depricated) not recognized")

    print(f"Running {args.scenario} mode for {args.model}")


    run_name = f"{args.model}_{args.scenario}_{args.dataset}"
    if args.scenario == "commoncorruption":
        run_name = f"{args.model}_{args.scenario}_{args.dataset}_{args.commoncorruption}_{args.severity}"
    elif args.scenario == "attack":
        run_name = f"{args.model}_{args.scenario}_{args.dataset}_{args.attack_type}"

    mlflow.set_tag("mlflow.runName", run_name)

        
    if args.scenario == "train":
        print("Started train")
        main.train()

    elif args.scenario == "test" or args.scenario == "commoncorruption":
        main.test()

    elif args.scenario == "attack":
        main.attack(attack_type=args.attack_type)

    else:
        raise ValueError("Scenario not recognized")
