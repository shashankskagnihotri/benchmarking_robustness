import argparse
import os
import sys
import mlflow

mlflow.set_tracking_uri("/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/mlflow/")
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Specify an architecture', choices=['cfnet', 'gwcnet', 'psmnet', 'sttr', 'sttr-light'])
parser.add_argument('--scenario', required=True, help='Specify whether to train or test the model', choices=['train', 'test', 'attack', 'commoncorruption'])
parser.add_argument('--commoncorruption', required=False, help='Specify the name of the common corruptions to apply. --phase must be test')
parser.add_argument('--severity', required=False, help='Specify the severity level of the common corruptions to apply. --phase must be test and --commoncorruption must be specified')
parser.add_argument('--attack_type', required=False, help='Specify the attack to apply. --phase must be test')

args, unknown = parser.parse_known_args()

if args.scenario == "commoncorruption" and (args.commoncorruption is None or args.severity is None):
    raise ValueError("If --mode is commoncorruption, --commoncorruption and --severity must be specified")

if args.scenario == "attack" and args.attack_type is None:
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
    raise ValueError("If --scenario is attack, --attack must be specified")
=======
>>>>>>> Stashed changes
    raise ValueError("If --mode is attack, --attack must be specified")

# Add the CFN#
cfnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CFNet'))
sys.path.append(cfnet_path)
<<<<<<< Updated upstream
=======
>>>>>>> disparity_estimation/feature/wrapper
>>>>>>> Stashed changes

# https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
with mlflow.start_run(experiment_id='128987742873377588'):

    if args.model.lower() == "cfnet":
        from CFNet import main

<<<<<<< Updated upstream
    else:
        raise ValueError("Architecture not recognized")

    print(f"Running {args.scenario} mode for {args.model}")


    run_name = f"{args.model}_{args.scenario}"
    if args.scenario == "commoncorruption":
        run_name = f"{args.model}_{args.scenario}_{args.commoncorruption}_{args.severity}"
    elif args.scenario == "attack":
        run_name = f"{args.model}_{args.scenario}_{args.attack_type}"

    mlflow.set_tag("mlflow.runName", run_name)

=======
<<<<<<< HEAD
    print(f"Running {args.scenario} scenario for {args.architecture}")
>>>>>>> Stashed changes
        
    if args.scenario == "train":
        main.train()

    elif args.scenario == "test" or args.scenario == "commoncorruption":
        main.test()

    elif args.scenario == "attack":
        main.attack(attack_type=args.attack_type)

<<<<<<< Updated upstream
=======
        if args.commoncorruption not in args.unknown:
            raise ValueError("Common corruption not recognized")

        mlflow.set_tag("mlflow.runName", f"{args.scenario}_commoncorruption_{args.commoncorruption}_{args.severity}")
        main.test()
=======
    print(f"Running {args.scenario} mode for {args.model}")


    run_name = f"{args.model}_{args.scenario}"
    if args.scenario == "commoncorruption":
        run_name = f"{args.model}_{args.scenario}_{args.commoncorruption}_{args.severity}"
    elif args.scenario == "attack":
        run_name = f"{args.model}_{args.scenario}_{args.attack_type}"

    mlflow.set_tag("mlflow.runName", run_name)

        
    if args.scenario == "train":
        main.train()

    elif args.scenario == "test" or args.scenario == "commoncorruption":
        main.test()

    elif args.scenario == "attack":
        main.attack(attack_type=args.attack_type)

>>>>>>> disparity_estimation/feature/wrapper
>>>>>>> Stashed changes
    else:
        raise ValueError("Mode not recognized")