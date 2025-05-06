import os
import argparse
import yaml
import pandas as pd
import torch

from dataloader.utils import get_checkpoint_path
from .wrapper import run_from_args

common_corruptions = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
] 
    
attacks = [
    'fgsm',
    'pgd',
    'apgd',
    'bim',
    'cospgd',
]


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sttr', help='Specify an model', choices=['cfnet', 'gwcnet-g', 'gwcnet-gc', 'sttr', 'sttr-light'])
    parser.add_argument('--scenario', type=str, default='test', help='Specify whether to train or test the model', choices=['train', 'test', 'attack', 'commoncorruption'])
    parser.add_argument('--dataset', type=str, default='kitti2015', help='Specify the dataset to use', choices=['sceneflow', 'kitti2015'])
    parser.add_argument('--commoncorruption', type=str, default=None, help='Specify the name of the common corruptions to apply')
    parser.add_argument('--severity', type=int, default=None, help='Specify the severity level of the common corruptions to apply')
    parser.add_argument('--attack_type', type=str, default=None, help='Specify the attack to apply')
    parser.add_argument('--epsilon', type=float, default=8/255, help='Specify the permissible perturbation budget')
    parser.add_argument('--alpha', type=float, default=0.01, help='Specify the step size of the attack')
    parser.add_argument('--num_iterations', type=int, default=20, help='Specify the number of attack iterations')
    parser.add_argument('--norm', type=str, default='Linf', help='Specify the lp-norm to be used for bounding the perturbation', choices=['Linf', 'L2'])
    parser.add_argument('--experiment', type=str, default='debug', choices=['debug', 'Common_Corruptions'], help='Specify the experiment to log to')
    parser.add_argument('--loadckpt', type=str, default=None, help='Specify the checkpoint to load')
    parser.add_argument('--dataset_path', type=str, default=None, help='Specify the path to the dataset')
    parser.add_argument("--batch_size", type=int, default=4, help="Specify the training batch size")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Specify the testing batch size")
    
    args, unknown = parser.parse_known_args()
    return args


def load_model(model_name: str, dataset: str):
    model_name = model_name.lower()
    dataset = dataset.lower()

    checkpoint = get_checkpoint_path(dataset, model_name)
    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint {checkpoint} does not exist")
    
    if model_name == "cfnet":
        from CFNet.models import __models__
        model = torch.nn.DataParallel(__models__[model_name](256))
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["model"])
        print("Loaded CFNet")
    elif model_name in "gwcnet-gc":
        from GwcNet.models import __models__
        model = torch.nn.DataParallel(__models__[model_name](192))
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["model"])
        print("Loaded GWCNet")
    elif model_name == "sttr":
        from sttr.main import get_args_parser
        from sttr.module.sttr import STTR
        ap = argparse.ArgumentParser('', parents=[get_args_parser()])
        args = ap.parse_args()
        model = STTR(args)
        print("Loaded STTR")
    elif model_name == "sttr-light":
        from sttr_light.main import get_args_parser
        from sttr_light.module.sttr import STTR
        ap = argparse.ArgumentParser('', parents=[get_args_parser()])
        args = ap.parse_args()
        model = STTR(args)
        print("Loaded STTR-Light")
    return model


def evaluate(model_name: str, dataset: str, retrieve_existing: bool, threat_config: str):
    model_name = model_name.lower()
    dataset = dataset.lower()
    model = load_model(model_name, dataset)

    if not os.path.exists(threat_config):
        raise ValueError(f"Config {threat_config} does not exist")

    with open(threat_config, 'r') as f:
        config = yaml.safe_load(f)
    threat_model = config['threat_model'].lower()
    
    if threat_model in common_corruptions:
        severity = int(config['severity'])
        if severity < 1 or severity > 5:
            raise ValueError("Severity must be an integer between 1 and 5.")

        if retrieve_existing:
            csv_path = "eval_csv/eval_2d_corruptions.csv"
            df = pd.read_csv(csv_path)

            filtered= df[
                (df['architecture'].str.lower() == model_name) &
                (df['dataset'].str.lower() == dataset) &
                (df['severity'] == f"sev{severity}") &
                (df['corruption'].str.lower() == threat_model)
            ]
            
            if not filtered.empty:
                # found evaluation in the csv
                results = {'epe': None, 'iou': None, '3px_error': None}
                for metric in results.keys():
                    row = filtered[filtered['metric'] == metric]
                    if not row.empty:
                        results[metric] = float(row['value'].values[0])
                return model, results
        
        args = get_args_parser()
        args.model = model_name
        args.dataset = dataset
        args.scenario = 'commoncorruption'
        args.commoncorruption = threat_model
        args.severity = int(config['severity'])
        if args.model in ['sttr', 'sttr-light']:
            args.test_batch_size = 1
        results = run_from_args(args)
    elif threat_model in attacks:
        if retrieve_existing:
            csv_path = "eval_csv/eval_adv_attacks.csv"
            df = pd.read_csv(csv_path)

            filtered = df[
                (df['dataset'].str.lower() == dataset.lower()) &
                (df['Architecture'].str.lower() == model_name.lower()) &
                (df['Iteration'] == int(config['iterations']) - 1)
            ]

            # evaluation run from setting: epsilon = 8.0, alpha = 0.01, lp_norm = 'Linf'
            if not (filtered.empty or float(config['epsilon']) != 8.0 or float(config['alpha']) != 0.01 or config['lp_norm'] != 'Linf'):
                # found evaluation in the csv
                results = {'epe': None, 'iou': None, '3px_error': None}
                value = filtered.iloc[0][threat_model]
                if value != '-':
                    results['epe'] = float(value)
                return model, results
            
        args = get_args_parser()
        args.model = model_name
        args.dataset = dataset
        args.scenario = 'attack'
        args.attack_type = threat_model
        args.epsilon = float(config['epsilon']) / 255
        args.alpha = float(config['alpha'])
        args.num_iterations = int(config['iterations'])
        args.norm = config['lp_norm']
        args.test_batch_size = 1
        results = run_from_args(args)
    else:
        ValueError(f"Threat model {config['threat_model']} is not supported")

    return model, results
    

def main():
    """
    model = load_model(model_name='STTR', dataset='SceneFlow')
    model = load_model(model_name='STTR-Light', dataset='SceneFlow')
    model = load_model(model_name='CFNet', dataset='SceneFlow')
    model = load_model(model_name='GWCNet-GC', dataset='SceneFlow')
    model = load_model(model_name='GWCNet-G', dataset='SceneFlow')
    model = load_model(model_name='GWCNet-G', dataset='KITTI2015')
    model = load_model(model_name='STTR', dataset='KITTI2015')
    """
    model, results = evaluate(
        model_name='CFNet', # STTR-Light, CFNet
        dataset='SceneFlow',
        retrieve_existing=False,
        threat_config='configs/adv_attacks.yml',
    )
    print(results)
    """
    model, results = evaluate(
        model_name='CFNet',#, , STTR-Light, CFNet
        dataset='SceneFlow',
        retrieve_existing=False,
        threat_config='configs/2d_corruptions.yml',
    )
    print(results)
    """


if __name__ == '__main__':
    main()