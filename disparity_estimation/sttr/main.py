#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import argparse
import os
import random

import numpy as np
import torch

# from .dataset import build_data_loader
from .module.sttr import STTR
from .utilities.checkpoint_saver import Saver
from .utilities.eval import evaluate
from .utilities.inference import inference
from .utilities.summary_logger import TensorboardSummary
from .utilities.train import train_one_epoch
from .utilities.foward_pass import set_downsample, forward_pass
from .module.loss import build_criterion

from .dataloader import get_data_loader_1
from disparity_estimation.attacks.attack import CosPGDAttack, FGSMAttack, PGDAttack, APGDAttack, BIMAttack


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_regression', default=2e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--loadckpt', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action='store_true', help='load model from checkpoint, but discard optimizer state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--downsample', default=3, type=int, help='Ratio to downsample width/height')
    parser.add_argument('--apex', action='store_true', help='enable mixed precision training')

    # * STTR
    parser.add_argument('--channel_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    # * Positional Encoding
    parser.add_argument('--position_encoding', default='sine1d_rel', type=str, choices=('sine1d_rel', 'none'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_attn_layers', default=6, type=int, help="Number of attention layers in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * Regression Head
    parser.add_argument('--regression_head', default='ot', type=str, choices=('softmax', 'ot'),
                        help='Normalization to be used')
    parser.add_argument('--context_adjustment_layer', default='cal', choices=['cal', 'none'], type=str)
    parser.add_argument('--cal_num_blocks', default=8, type=int)
    parser.add_argument('--cal_feat_dim', default=16, type=int)
    parser.add_argument('--cal_expansion_ratio', default=4, type=int)

    # * Dataset parameters
    parser.add_argument('--dataset', default='sceneflow', type=str, help='dataset to train/eval on')
    parser.add_argument('--datapath', default='', type=str, help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')

    # * Loss
    parser.add_argument('--px_error_threshold', type=int, default=3,
                        help='Number of pixels for error computation (default 3 px)')
    parser.add_argument('--loss_weight', type=str, default='rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0',
                        help='Weight for losses')
    parser.add_argument('--validation_max_disp', type=int, default=-1)

    # attack 
    parser.add_argument('--attack_type', required=False, default=None, help='Specify the attack to apply. --phase must be test')

    return parser


def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best, amp=None):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)


def print_param(model):
    """
    print number of parameters in the model
    """

    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad)
    print('number of params in backbone:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if
                       'transformer' in n and 'regression' not in n and p.requires_grad)
    print('number of params in transformer:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'tokenizer' in n and p.requires_grad)

    print('number of params in tokenizer:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'regression' in n and p.requires_grad)
    print('number of params in regression:', f'{n_parameters:,}')


def main(args):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = STTR(args).to(device)
    print_param(model)

    # set learning rate
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if
                    "backbone" not in n and "regression" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "regression" in n and p.requires_grad],
            "lr": args.lr_regression,
        },
    ]

    # define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    # mixed precision training
    print("AMP: ", args.apex)
    if args.apex:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else:
        amp = None

    # load checkpoint if provided
    prev_best = np.inf
    if args.loadckpt != '':
        if not os.path.isfile(args.loadckpt):
            raise RuntimeError(f"=> no checkpoint found at '{args.loadckpt}'")
        checkpoint = torch.load(args.loadckpt)

        pretrained_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        # check missing and unexpected keys
        if len(missing) > 0:
            print("Missing keys: ", ','.join(missing))
            raise Exception("Missing keys.")
        unexpected_filtered = [k for k in unexpected if
                               'running_mean' not in k and 'running_var' not in k]  # skip bn params
        if len(unexpected_filtered) > 0:
            print("Unexpected keys: ", ','.join(unexpected_filtered))
            raise Exception("Unexpected keys.")
        print("Pre-trained model successfully loaded.")

        # if not ft/inference/eval, load states for optimizer, lr_scheduler, amp and prev best
        if not (args.ft or args.inference or args.eval or args.attack_type):
            if len(unexpected) > 0:  # loaded checkpoint has bn parameters, legacy resume, skip loading
                raise Exception("Resuming legacy model with BN parameters. Not possible due to BN param change. " +
                                "Do you want to finetune or inference? If so, check your arguments.")
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                prev_best = checkpoint['best_pred']
                if args.apex:
                    amp.load_state_dict(checkpoint['amp'])
                print("Pre-trained optimizer, lr scheduler and stats successfully loaded.")

    # inference
    if args.inference:
        print("Start inference")
        data_loader_train, data_loader_val, data_loader_test = get_data_loader_1(args, "sttr")
        inference(model, data_loader_test, device, args.downsample)

        return

    # initiate saver and logger
    checkpoint_saver = Saver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)

    # build dataloader
    # data_loader_train, data_loader_val, data_loader_test = build_data_loader(args)
    data_loader_train, data_loader_val, data_loader_test = get_data_loader_1(args, "sttr")
    # print("ADEBUG: val SIZE" + str(len(data_loader_val)))
    # print("ADEBUG: train SIZE" + str(len(data_loader_train)))
    # print("ADEBUG: test SIZE" + str(len(data_loader_test)))

    # build loss criterion
    criterion = build_criterion(args)

    # set downsample rate
    set_downsample(args)
    
    # eval
    if args.eval:
        eval_stats = evaluate(model, criterion, data_loader_test, device, 0, summary_writer, save_output=False)
        results = {'epe': eval_stats['epe'], 'iou': eval_stats['iou'], '3px_error': eval_stats['px_error_rate']}
        return results
    
    if args.attack_type is not None:
        # mixed precision training
        scaler = torch.cuda.amp.GradScaler() if args.apex else None

        # model = torch.nn.DataParallel(model.half())
        # model = model.half().to('cuda:0')
        # Annahme: model ist dein geladenes Modell
        # for name, param in model.named_parameters():
        #    print(f"Parameter: {name}, Datentyp: {param.dtype}, Device: {param.device}")
        model.eval()
        criterion.eval()
        stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0, 'total_px': 0.0}

        if args.attack_type == "cospgd":
            attacker = CosPGDAttack(
                model, architecture=args.model, epsilon=args.epsilon, alpha=args.alpha, num_iterations=args.num_iterations, 
                norm=args.norm, num_classes=None, targeted=False, criterion=criterion, stats=stats, device=device,
            )
        elif args.attack_type == "fgsm":
            attacker = FGSMAttack(
                model, architecture=args.model, epsilon=args.epsilon, targeted=False, criterion=criterion, stats=stats,
            ) 
        elif args.attack_type == "pgd":
            attacker = PGDAttack(
                model, architecture=args.model, epsilon=args.epsilon, alpha=args.alpha, num_iterations=args.num_iterations, 
                norm=args.norm, random_start=True, targeted=False, criterion=criterion, stats=stats, device=device,
            )
        elif args.attack_type =='bim':
            attacker = BIMAttack(
                model, architecture=args.model, epsilon=args.epsilon, alpha=args.alpha, num_iterations=args.num_iterations, 
                norm=args.norm, targeted=False, criterion=criterion, stats=stats
            ) 
        elif args.attack_type == 'apgd':
            attacker = APGDAttack(
                model, architecture=args.model, eps=args.epsilon, num_iterations=args.num_iterations, 
                norm=args.norm, criterion=criterion, stats=stats, device=device,
            )
        else:
            raise ValueError("Attack type not recognized")

        eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0, 'total_px': 0.0}
        valid_samples = len(data_loader_test)

        for batch_idx, sample in enumerate(data_loader_test):
            print("batch", batch_idx)
            perturbed_results = attacker.attack(sample["left"], sample["right"], sample["disp"], sample["occ_mask"], sample["occ_mask_right"])
            for iteration in perturbed_results.keys():
                model.eval()
                perturbed_left, perturbed_right = perturbed_results[iteration]

                # forward pass
                data = {
                    'left': perturbed_left,
                    'right': perturbed_right,
                    'disp': sample['disp'],
                    'occ_mask': sample['occ_mask'],
                    'occ_mask_right': sample['occ_mask_right'],
                }
                outputs, losses, sampled_disp = forward_pass(model, data, device, criterion, eval_stats)

                print(f"Iteration {iteration} loss: {losses['aggregated'].item()}")
 
                if losses is None:
                    valid_samples -= 1
                    continue
 
                # clear cache
                torch.cuda.empty_cache()
 
        # compute avg
        eval_stats['epe'] = eval_stats['epe'] / (valid_samples * len(perturbed_results.keys()))
        eval_stats['iou'] = eval_stats['iou'] / (valid_samples * len(perturbed_results.keys()))
        eval_stats['px_error_rate'] = eval_stats['error_px'] / eval_stats['total_px']
        results = {'epe': eval_stats['epe'], 'iou': eval_stats['iou'], '3px_error': eval_stats['px_error_rate']}
        return results

    # train
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # train
        print("Epoch: %d" % epoch)
        train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, summary_writer,
                        args.clip_max_norm, amp)

        # step lr if not pretraining
        if not args.pre_train:
            lr_scheduler.step()
            print("current learning rate", lr_scheduler.get_lr())

        # empty cache
        torch.cuda.empty_cache()

        # save if pretrain, save every 50 epochs
        if args.pre_train or epoch % 50 == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, amp)

        # validate
        eval_stats = evaluate(model, criterion, data_loader_val, device, epoch, summary_writer, False)
        # save if best
        if prev_best > eval_stats['epe'] and 0.5 > eval_stats['px_error_rate']:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, True, amp)

    # save final model
    save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, amp)

    return


def merge_args(base_args: argparse.Namespace, override_args: argparse.Namespace) -> argparse.Namespace:
    merged = vars(base_args).copy()
    merged.update(vars(override_args))
    return argparse.Namespace(**merged)


def train(args_from_wrapper=None):
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args, unknown = ap.parse_known_args()
    if args_from_wrapper is not None:
        args = merge_args(args, args_from_wrapper)
    args.eval = False
    return main(args)


def test(args_from_wrapper=None):
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args, unknown = ap.parse_known_args()
    if args_from_wrapper is not None:
        args = merge_args(args, args_from_wrapper)
    args.eval = True
    return main(args)
    

def attack(args_from_wrapper=None):
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args, unknown = ap.parse_known_args()
    if args_from_wrapper is not None:
        args = merge_args(args, args_from_wrapper)
    return main(args) 


# if __name__ == '__main__':
#     ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
#     args_ = ap.parse_known_args()
#     main(args_)