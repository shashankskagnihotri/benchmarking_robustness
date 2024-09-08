from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import mlflow

# from dataloader import listflowfile as lt
from dataloader import get_dataset
from torch.utils.data import DataLoader
from models import *
from dataloader.summary_logger import TensorboardSummary

parser = argparse.ArgumentParser(description="PSMNet")
parser.add_argument("--dataset", required=True, help="dataset name")
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument("--model", default="stackhourglass", help="select model")
parser.add_argument("--datapath", required=True, help="datapath")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
parser.add_argument("--loadmodel", default=None, help="load model")
parser.add_argument("--savemodel", default="./", help="save model")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--eval", action="store_true", default=False, help="evaluate model on test set"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

# TrainImgLoader = torch.utils.data.DataLoader(
#          SceneFlowFlyingThings3DDataset(all_left_img,all_right_img,all_left_disp, True),
#          batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

# TestImgLoader = torch.utils.data.DataLoader(
#          SceneFlowFlyingThings3DDataset(test_left_img,test_right_img,test_left_disp, False),
#          batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)

logger_path = "./runs"
logger = TensorboardSummary(logger_path).config_logger(0)

train_dataset = get_dataset(
    args.dataset, args.datapath, architeture_name="PSMNet", split="Train"
)
test_dataset = get_dataset(
    args.dataset, args.datapath, architeture_name="PSMNet", split="Test"
)

TrainImgLoader = DataLoader(
    train_dataset, batch_size=12, shuffle=True, num_workers=8, drop_last=True
)
TestImgLoader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False
)


if args.model == "stackhourglass":
    model = stackhourglass(args.maxdisp)
elif args.model == "basic":
    model = basic(args.maxdisp)
else:
    print("no model")

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

if args.loadmodel is not None:
    print("Load pretrained model")
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict["state_dict"])
    if "optimizer_state_dict" in pretrain_dict:
        optimizer.load_state_dict(pretrain_dict["optimizer_state_dict"])

print(
    "Number of model parameters: {}".format(
        sum([p.data.nelement() for p in model.parameters()])
    )
)


def train(imgL, imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == "stackhourglass":
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = (
            0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True)
            + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True)
            + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
        )
    elif args.model == "basic":
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data


def test(imgL, imgR, disp_true):

    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    mask = disp_true < 192
    # ----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = F.l1_loss(
            img[mask], disp_true[mask]
        )  # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

    return loss.data.cpu()


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print(f"This is {epoch}-th epoch")
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print(
                "Iter %d training loss = %.3f , time = %.2f"
                % (batch_idx, loss, time.time() - start_time)
            )
            total_train_loss += loss
        print(
            "epoch %d total training loss = %.3f"
            % (epoch, total_train_loss / len(TrainImgLoader))
        )

        # INFERENCE
        total_test_loss = inference(TestImgLoader)

        # SAVE
        savefilename = args.savemodel + "/checkpoint_" + str(epoch) + ".tar"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "train_loss": total_train_loss / len(TrainImgLoader),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            savefilename,
        )

    print("full training time = %.2f HR" % ((time.time() - start_full_time) / 3600))

    # ----------------------------------------------------------------------------------
    # SAVE test information
    savefilename = args.savemodel + "testinformation.tar"
    torch.save(
        {
            "test_loss": total_test_loss / len(TestImgLoader),
        },
        savefilename,
    )


def inference(TestImgLoader: DataLoader):
    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L)
        print("Iter %d test loss = %.3f" % (batch_idx, test_loss))
        total_test_loss += test_loss

    print("total test loss = %.3f" % (total_test_loss / len(TestImgLoader)))
    return total_test_loss


def attack(attack_type: str):

    from attacks import CosPGDAttack, FGSMAttack, PGDAttack, APGDAttack,BIMAttack

    epsilon = 0.03
    alpha = 0.01
    num_iterations = 20
    norm = "Linf" 
    

    if attack_type == "cospgd":
        attacker = CosPGDAttack(model, epsilon, alpha, num_iterations, norm,num_classes=None, targeted=False )
    elif attack_type == "fgsm":
        attacker = FGSMAttack( model, epsilon, targeted=False) 

    elif attack_type == "pgd":
        attacker = PGDAttack(model,epsilon,num_iterations,alpha,norm,random_start=True,targeted=False)

    elif attack_type =='bim':
        attacker = BIMAttack(model,epsilon,num_iterations,alpha,norm, targeted=False) 
        
    elif attack_type == 'apgd':
        attacker = APGDAttack(model, num_iterations,norm, epsilon)
    
    else:
        raise ValueError("Attack type not recognized")

    for batch_idx, sample in enumerate(TestImgLoader):
        perturbed_results = attacker.attack(sample["left"], sample["right"], sample["disparity"])
        for iteration in perturbed_results.keys():
            model.eval()
            perturbed_left, perturbed_right = perturbed_results[iteration]
            loss, scalar_outputs, image_outputs  = test_sample({'left':perturbed_left,'right':perturbed_right,'disparity':sample["disparity"]})
            save_scalars(logger, "test", scalar_outputs, batch_idx)


        print("batch", batch_idx)


if __name__ == "__main__":
    if args.eval:
        inference(TestImgLoader)
    else:
        main()
