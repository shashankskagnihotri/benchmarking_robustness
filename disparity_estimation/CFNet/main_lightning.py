import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow

from models import __models__, model_loss
from utils import *
from dataloader import get_dataset

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', required=True, help='dataset name')
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--fast_dev_run', action='store_true', help='run only a few batches for testing')

args, _ = parser.parse_known_args()

class CFNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = __models__[args.model](args.maxdisp)
        self.save_hyperparameters()

    def forward(self, imgL, imgR):
        return self.model(imgL, imgR)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    # def _step(self, batch, stage):
    #     imgL, imgR, disp_gt = batch['left'], batch['right'], batch['disparity']
    #     disp_ests = self(imgL, imgR)
    #     mask = (disp_gt < self.args.maxdisp) & (disp_gt > 0)
    #     loss = model_loss(disp_ests, disp_gt, mask)

    #     self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    #     scalar_outputs = {"loss": loss}
    def _step(self, batch, stage):
        imgL_batch, imgR_batch, disp_gt_batch = batch['left'], batch['right'], batch['disparity']
        total_loss = 0
        for imgL, imgR, disp_gt in zip(imgL_batch, imgR_batch, disp_gt_batch):
            imgL = imgL.unsqueeze(0)  # Add batch dimension
            imgR = imgR.unsqueeze(0)  # Add batch dimension
            disp_gt = disp_gt.unsqueeze(0)  # Add batch dimension
            
            disp_ests = self(imgL, imgR)
            mask = (disp_gt < self.args.maxdisp) & (disp_gt > 0)
            loss = model_loss(disp_ests, disp_gt, mask)
            
            total_loss += loss
        
        avg_loss = total_loss / len(imgL_batch)
        
        self.log(f"{stage}_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        scalar_outputs = {"loss": avg_loss}
        scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

        for key, value in scalar_outputs.items():
            if isinstance(value, list):
                self.log(f"{stage}_{key}", value[-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            else:
                self.log(f"{stage}_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in self.args.lrepochs.split(':')[0].split(',')], 
                                                   gamma=float(self.args.lrepochs.split(':')[1]))
        return [optimizer], [scheduler]

def main():
    pl.seed_everything(args.seed)
    
    train_dataset = get_dataset(args.dataset, args.datapath, architeture_name="CFNet", split='train')
    test_dataset  = get_dataset(args.dataset, args.datapath, architeture_name="CFNet", split='test')

    if 'kitti' in args.dataset.lower():
        val_size = int(0.2 * len(train_dataset))
        test_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size - test_size
        train_subset, val_subset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])
    else:
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    if args.fast_dev_run:
        fast_dev_run_size = 10
        train_subset = Subset(train_subset, list(range(fast_dev_run_size)))
        val_subset = Subset(val_subset, list(range(fast_dev_run_size)))
        test_dataset = Subset(test_dataset, list(range(fast_dev_run_size)))

    train_loader = DataLoader(train_subset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_subset, args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Debug print to check data loading
    for i, batch in enumerate(train_loader):
        print(f"Train loader batch {i} - left shape: {batch['left'].shape}, right shape: {batch['right'].shape}, disparity shape: {batch['disparity'].shape}")
        if i >= 2:
            break

    for i, batch in enumerate(val_loader):
        print(f"Validation loader batch {i} - left shape: {batch['left'].shape}, right shape: {batch['right'].shape}, disparity shape: {batch['disparity'].shape}")
        if i >= 2:
            break

    for i, batch in enumerate(test_loader):
        print(f"Test loader batch {i} - left shape: {batch['left'].shape}, right shape: {batch['right'].shape}, disparity shape: {batch['disparity'].shape}")
        if i >= 2:
            break

    model = CFNetLightning(args)

    logger = TensorBoardLogger(save_dir=args.logdir, name="cfnet_logs")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.logdir,
        filename='cfnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        every_n_epochs=1
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        fast_dev_run=args.fast_dev_run
    )

    mlflow.pytorch.autolog()

    trainer.fit(model, train_loader, val_loader)#, ckpt_path=args.loadckpt if args.resume else None)
    trainer.test(model, test_loader)

if __name__ == '__main__':
    main()