import argparse
import os
import sys
import mlflow
from .dataloader.utils import get_checkpoint_path, get_dataset_path


def run_from_args(args):
    mlflow.set_tracking_uri('mlflows')

    args.scenario = args.scenario.lower()
    args.model = args.model.lower()

    if args.scenario == "commoncorruption" and (args.commoncorruption is None or args.severity is None):
        raise ValueError("If --scenario is commoncorruption, --commoncorruption and --severity must be specified")

    if args.scenario == "attack" and args.attack_type is None:
        raise ValueError("If --scenario is attack, --attack must be specified")

    if args.loadckpt is None:
        args.loadckpt = get_checkpoint_path(args.dataset, args.model)

    if args.loadckpt is not None and not os.path.exists(args.loadckpt):
        raise ValueError(f"Checkpoint {args.loadckpt} does not exist")

    if args.dataset_path is None:
        if args.dataset is None:
            raise ValueError("Either --dataset_path or --dataset must be specified")
        elif args.scenario == "commoncorruption":
            args.dataset_path = get_dataset_path(args.dataset, args.commoncorruption, args.severity)
        else:
            args.dataset_path = get_dataset_path(args.dataset)            

    if args.dataset_path is not None and not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset path {args.dataset_path} does not exist")

    experiment_name = f"{args.experiment}_{args.model}_{args.dataset}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment ID: {experiment_id}")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(vars(args))

        if args.model == "cfnet":
            from .CFNet import main
        elif args.model in "gwcnet-gc":
            from .GwcNet import main
        elif args.model == "sttr":
            from .sttr import main
        elif args.model == "sttr-light":
            from .sttr_light import main
        else:
            raise ValueError("Architecture/Model not recognized")

        run_name = f"{args.model}_{args.scenario}_{args.dataset}"
        if args.scenario == "commoncorruption":
            run_name = f"{args.model}_{args.scenario}_{args.dataset}_{args.commoncorruption}_{args.severity}"
        elif args.scenario == "attack":
            run_name = f"{args.model}_{args.scenario}_{args.dataset}_{args.attack_type}"

        mlflow.set_tag("mlflow.runName", run_name)
        
        if args.scenario == "train":
            return main.train(args)
        elif args.scenario == "test" or args.scenario == "commoncorruption":
            return main.test(args)
        elif args.scenario == "attack":
            return main.attack(args)
        else:
            raise ValueError("Scenario not recognized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Specify an model', choices=['cfnet', 'gwcnet', 'gwcnet-g', 'gwcnet-gc', 'psmnet', 'sttr', 'sttr-light'])
    parser.add_argument('--scenario', required=True, help='Specify whether to train or test the model', choices=['train', 'test', 'attack', 'commoncorruption'])
    parser.add_argument('--dataset', required=True, help='Specify the dataset to use', choices=['sceneflow', 'mpisintel', 'kitti', 'kitti2015', 'eth3d', 'mpisintel'])
    parser.add_argument('--commoncorruption', required=False, help='Specify the name of the common corruptions to apply. --phase must be test')
    parser.add_argument('--severity', required=False, type=int, help='Specify the severity level of the common corruptions to apply. --phase must be test and --commoncorruption must be specified')
    parser.add_argument('--attack_type', required=False, help='Specify the attack to apply. --phase must be test')
    parser.add_argument('--epsilon', required=False, default=8/255, type=float, help='Specify the permissible perturbation budget')
    parser.add_argument('--alpha', required=False, default=0.01, type=float, help='Specify the step size of the attack')
    parser.add_argument('--num_iterations', required=False, default=20, type=int, help='Specify the number of attack iterations')
    parser.add_argument('--norm', required=False, default='Linf', type=str, help='Specify the lp-norm to be used for bounding the perturbation', choices=['Linf', 'L2'])
    parser.add_argument('--experiment', required=False, default='debug', type=str, choices=['debug', 'Common_Corruptions'], help='Specify the experiment to log to')
    parser.add_argument('--loadckpt', required=False, default=None, type=str, help='Specify the checkpoint to load')
    parser.add_argument('--dataset_path', required=False, default=None, type=str, help='Specify the path to the dataset')
    parser.add_argument("--batch_size", required=False, type=int, default=4, help="Specify the training batch size")
    parser.add_argument("--test_batch_size", required=False, type=int, default=1, help="Specify the testing batch size")
    
    args, unknown = parser.parse_known_args()
    run_from_args(args)