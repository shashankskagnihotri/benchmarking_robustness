from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

# from datasets import __datasets__
from .models import __models__, model_loss
from .utils import *
from torch.utils.data import DataLoader, random_split, Subset
import gc
import mlflow
from torchvision import transforms

from .dataloader import get_data_loader_1
from attacks.attack import CosPGDAttack, FGSMAttack, PGDAttack, APGDAttack, BIMAttack


cudnn.benchmark = True


def get_args_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(
        description="Group-wise Correlation Stereo Network (GwcNet)"
    )
    
    parser.add_argument("--model", default="gwcnet-g", help="select a model structure", choices=__models__.keys())
    parser.add_argument("--maxdisp", type=int, default=192, help="maximum disparity")

    parser.add_argument("--dataset", type=str, default="sceneflow", help="dataset name")
    parser.add_argument("--datapath", type=str, default="", help="data path")
    parser.add_argument("--lr", type=float, default=0.001, help="base learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
    parser.add_argument("--test_batch_size", type=int, default=4, help="testing batch size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument("--lrepochs", type=str, default="200:10", help="the epochs to decay lr: the downscale rate",)

    parser.add_argument("--logdir", type=str, default="dev", help="the directory to save logs and checkpoints")
    parser.add_argument("--loadckpt", help="load the weights from a specific checkpoint")
    parser.add_argument("--resume", action="store_true", help="continue training the model")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    parser.add_argument("--summary_freq", type=int, default=20, help="the frequency of saving summary")
    parser.add_argument("--save_freq", type=int, default=1, help="the frequency of saving checkpoint")

    # parse arguments, set seeds
    args, _ = parser.parse_known_args()
    return args


def merge_args(base_args: argparse.Namespace, override_args: argparse.Namespace) -> argparse.Namespace:
    merged = vars(base_args).copy()
    merged.update(vars(override_args))
    return argparse.Namespace(**merged)


def setup_environment(args):
    mlflow.log_params(vars(args))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)

    # create summary logger
    logger = SummaryWriter(args.logdir)

    # prepare Data
    TrainImgLoader, ValImgLoader, TestImgLoader = get_data_loader_1(args, "gwcnet-g") # TODO: Refine args here with architecture and model

    # model, optimizer
    # print(args.model)
    # print(__models__)
    model = __models__[args.model](args.maxdisp)
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # load parameters
    start_epoch = 0
    if args.resume:
        # find all checkpoints file and sort according to epoch id
        all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        all_saved_ckpts = sorted(
            all_saved_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
        print("loading the lastest model in logdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict["epoch"] + 1
    elif args.loadckpt:
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict["model"])
    print("start at epoch {}".format(start_epoch))
    return model, optimizer, logger, TrainImgLoader, ValImgLoader, TestImgLoader, start_epoch


def train(args_from_wrapper=None):
    args = get_args_parser()
    if args_from_wrapper is not None:
        args = merge_args(args, args_from_wrapper)
    model, optimizer, logger, TrainImgLoader, ValImgLoader, TestImgLoader, start_epoch = setup_environment(args)

    best_val_loss = -1
    bestepoch = 0
    error = 100
    print("Batch size: ", TestImgLoader.batch_size)
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(
                model, optimizer, sample, args.maxdisp, compute_metrics=do_summary,
            )
            if do_summary:
                save_scalars(logger, "train", scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                "Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}".format(
                    epoch_idx,
                    args.epochs,
                    batch_idx,
                    len(TrainImgLoader),
                    loss,
                    time.time() - start_time,
                )
            )
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "loss": loss,
            }
            torch.save(
                checkpoint_data,
                "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx),
            )
        gc.collect()

        # --- validation
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = len(ValImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(
                model, sample, args.maxdisp, compute_metrics=do_summary,
            )
            if do_summary:
                save_scalars(logger, "val", scalar_outputs, global_step)
                # save_images(logger, 'val', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print(
                "Epoch {}/{}, Iter {}/{}, val loss = {:.3f}, time = {:3f}".format(
                    epoch_idx,
                    args.epochs,
                    batch_idx,
                    len(TestImgLoader),
                    loss,
                    time.time() - start_time,
                )
            )
        # save checkpoint for lowest loss
        if loss < best_val_loss:
            checkpoint_data = {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "loss": loss,
            }
            torch.save(
                checkpoint_data,
                "{}/checkpoint_{:0>6}_best.ckpt".format(args.logdir, epoch_idx),
            )
            best_val_loss = loss

        gc.collect()
        # --- validation - END

        # testing
        avg_val_scalars = AverageMeterDict()
        # bestepoch = 0
        # error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(
                model, sample, args.maxdisp, compute_metrics=do_summary,
            )
            if do_summary:
                save_scalars(logger, "test", scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
            avg_val_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print(
                "Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
                    epoch_idx,
                    args.epochs,
                    batch_idx,
                    len(TestImgLoader),
                    loss,
                    time.time() - start_time,
                )
            )
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if nowerror < error:
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(
            logger, "fulltest", avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1)
        )
        print("avg_test_scalars", avg_test_scalars)
        print("MAX epoch %d total test error = %.5f" % (bestepoch, error))
        gc.collect()
    print("MAX epoch %d total test error = %.5f" % (bestepoch, error))


def test(args_from_wrapper=None):
    args = get_args_parser()
    if args_from_wrapper is not None:
        args = merge_args(args, args_from_wrapper)
    model, optimizer, logger, TrainImgLoader, ValImgLoader, TestImgLoader, start_epoch = setup_environment(args)

    # testing
    avg_test_scalars = AverageMeterDict()

    print(len(TestImgLoader))
    
    # for img_path_left, img_path_right in zip(TestImgLoader.dataset.img_left_filenames, TestImgLoader.dataset.img_right_filenames):
    #     try:
    #         TestImgLoader.dataset.load_image(img_path_left)
    #         TestImgLoader.dataset.load_image(img_path_right)

    #     except:
    #         print(img_path_left)
    #         print(img_path_right)
    #         print()

    # return 

    for batch_idx, sample in enumerate(TestImgLoader):
        global_step = len(TestImgLoader) * batch_idx
        start_time = time.time()
        do_summary = global_step % args.summary_freq == 0
        loss, scalar_outputs, image_outputs = test_sample(
            model, sample, args.maxdisp, compute_metrics=do_summary,
        )
        if do_summary:
            save_scalars(logger, "test", scalar_outputs, batch_idx)
            # save_images(logger, 'test', image_outputs, global_step)
            print(batch_idx, scalar_outputs)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print(
            "Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
                batch_idx, len(TestImgLoader), loss, time.time() - start_time
            )
        )
    avg_test_scalars = avg_test_scalars.mean()
    save_scalars(
        logger, "fulltest", avg_test_scalars, len(TestImgLoader) * (batch_idx + 1)
    )
    results = {'epe': avg_test_scalars['EPE'][0], 'iou': None, '3px_error': avg_test_scalars['Thres3'][0]}
    return results


# train one sample
def train_sample(model, optimizer, sample, maxdisp, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample["left"], sample["right"], sample["disparity"]
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {
        "disp_est": disp_ests,
        "disp_gt": disp_gt,
        "imgL": imgL,
        "imgR": imgR,
    }
    if compute_metrics:
        with torch.no_grad():
            image_disp_error = DispErrorImageFunc()
            image_outputs["errormap"] = [
                image_disp_error.apply(disp_est, disp_gt) for disp_est in disp_ests
            ]
            scalar_outputs["EPE"] = [
                EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests
            ]
            scalar_outputs["D1"] = [
                D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests
            ]
            scalar_outputs["Thres1"] = [
                Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests
            ]
            scalar_outputs["Thres2"] = [
                Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests
            ]
            scalar_outputs["Thres3"] = [
                Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests
            ]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(model, sample, maxdisp, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample["left"], sample["right"], sample["disparity"]
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {
        "disp_est": disp_ests,
        "disp_gt": disp_gt,
        "imgL": imgL,
        "imgR": imgR,
    }

    scalar_outputs["D1"] = [
        D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests
    ]
    scalar_outputs["EPE"] = [
        EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests
    ]
    scalar_outputs["Thres1"] = [
        Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests
    ]
    scalar_outputs["Thres2"] = [
        Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests
    ]
    scalar_outputs["Thres3"] = [
        Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests
    ]

    if compute_metrics:
        image_disp_error = DispErrorImageFunc()
        image_outputs["errormap"] = [
            image_disp_error.apply(disp_est, disp_gt) for disp_est in disp_ests
        ]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def attack(args_from_wrapper=None):
    args = get_args_parser()
    if args_from_wrapper is not None:
        args = merge_args(args, args_from_wrapper)
    model, optimizer, logger, TrainImgLoader, ValImgLoader, TestImgLoader, start_epoch = setup_environment(args)

    model.eval()
    model = NormalizedModel(model,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    num_iterations = 20

    if args.attack_type == "cospgd":
        attacker = CosPGDAttack(
            model, architecture=args.model, epsilon=args.epsilon, alpha=args.alpha, num_iterations=args.num_iterations, 
            norm=args.norm, num_classes=None, targeted=False,
        )
    elif args.attack_type == "fgsm":
        attacker = FGSMAttack(
            model, architecture=args.model, epsilon=args.epsilon, targeted=False,
        ) 
    elif args.attack_type == "pgd":
        attacker = PGDAttack(
            model, architecture=args.model, epsilon=args.epsilon, alpha=args.alpha, num_iterations=args.num_iterations, 
            norm=args.norm, random_start=True, targeted=False,
        )
    elif args.attack_type =='bim':
        attacker = BIMAttack(
            model, architecture=args.model, epsilon=args.epsilon, alpha=args.alpha, num_iterations=args.num_iterations, 
            norm=args.norm, targeted=False,
        ) 
    elif args.attack_type == 'apgd':
        attacker = APGDAttack(
            model, architecture=args.model, eps=args.epsilon, num_iterations=args.num_iterations, norm=args.norm,
        )
    else:
        raise ValueError("Attack type not recognized")
    
    avg_test_scalars = AverageMeterDict()

    for batch_idx, sample in enumerate(TestImgLoader):
        print("batch", batch_idx)
        # mean and std are the same for cfnet, gwcnet and sttr/sttr-light
        # model.eval()
        perturbed_results = attacker.attack(sample["left"], sample["right"], sample["disparity"])
        for iteration in perturbed_results.keys():
            model.eval()
            perturbed_left, perturbed_right = perturbed_results[iteration]
            loss, scalar_outputs, image_outputs  = test_sample(
                model, {'left':perturbed_left,'right':perturbed_right,'disparity':sample["disparity"]}, args.maxdisp,
            )
            save_scalars(logger, f"test_{iteration}", scalar_outputs, batch_idx)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print(f"Iteration {iteration} loss: {loss}")
            
        #     if batch_idx < 1:
        #         mlflow.log_image(perturbed_left.detach().squeeze().permute(1, 2, 0).cpu().numpy(), f"perturbed_left_batch_{batch_idx}_iteration_{iteration}.png")
        #         mlflow.log_image(perturbed_right.detach().squeeze().permute(1, 2, 0).cpu().numpy(), f"perturbed_right_{batch_idx}.png")
        #         mlflow.log_image(image_outputs["disp_est"][0].detach().squeeze().cpu().numpy(), f"disp_est_{batch_idx}_iteration_{iteration}.png")

        # # convert perturbed_images to an PIL.Image and store it in mlflow
        # if batch_idx < 10:
        #     print("Logging images")
        #     mlflow.log_image(perturbed_left.detach().squeeze().permute(1, 2, 0).cpu().numpy(), f"perturbed_left_{batch_idx}.png")
        #     mlflow.log_image(perturbed_right.detach().squeeze().permute(1, 2, 0).cpu().numpy(), f"perturbed_right_{batch_idx}.png")
        #     mlflow.log_image(image_outputs["disp_est"][0].detach().squeeze().cpu().numpy(), f"disp_est_{batch_idx}.png")
    
    avg_test_scalars = avg_test_scalars.mean()
    results = {'epe': avg_test_scalars['EPE'][0], 'iou': None, '3px_error': avg_test_scalars['Thres3'][0]}
    return results


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std) -> None:
        super(NormalizedModel, self).__init__()  
        self.mean = mean
        self.std = std
        self.model = model

    def forward(self, left, right, **kwargs):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        perturbed_left = normalize(left)
        perturbed_right = normalize(right)
        return self.model(perturbed_left, perturbed_right, **kwargs)
    

if __name__ == "__main__":
    train()
