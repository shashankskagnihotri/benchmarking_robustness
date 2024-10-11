#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
import mlflow

from utilities.misc import NestedTensor

downsample = 0


def set_downsample(args):
    global downsample
    downsample = args.downsample


def write_summary(stats, summary, step, mode):
    """
    write the current epoch result to tensorboard
    """
    mlflow.log_metric(mode + '/rr', stats['rr'], step)
    mlflow.log_metric(mode + '/l1', stats['l1'], step)
    mlflow.log_metric(mode + '/l1_raw', stats['l1_raw'], step)
    mlflow.log_metric(mode + '/occ_be', stats['occ_be'], step)
    mlflow.log_metric(mode + '/epe', stats['epe'], step)
    mlflow.log_metric(mode + '/iou', stats['iou'], step)
    mlflow.log_metric(mode + '/3px_error', stats['px_error_rate'], step)


def forward_pass(model, data, device, criterion, stats, idx=0, logger=None):
    """
    forward pass of the model given input
    """
    # TODO: Implement normalization here!
    model.to(device)
    
    # read data
    left, right = data['left'].to(device).half(), data['right'].to(device).half()
    disp, occ_mask, occ_mask_right = data['disp'].to(device).half(), data['occ_mask'].to(device).half(), \
                                     data['occ_mask_right'].to(device).half()

    # if need to downsample, sample with a provided stride
    bs, _, h, w = left.size()
    if downsample <= 0:
        sampled_cols = None
        sampled_rows = None
    else:
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).to(device)
        sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).to(device)

    # build the input
    inputs = NestedTensor(left, right, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp,
                          occ_mask=occ_mask, occ_mask_right=occ_mask_right)

    # forward pass
    outputs = model(inputs)

    # compute loss
    losses = criterion(inputs, outputs)

    if losses is None:
        return outputs, losses, disp

    # get the loss
    stats['rr'] += losses['rr'].item()
    stats['l1_raw'] += losses['l1_raw'].item()
    stats['l1'] += losses['l1'].item()
    stats['occ_be'] += losses['occ_be'].item()

    stats['iou'] += losses['iou'].item()
    stats['epe'] += losses['epe'].item()
    stats['error_px'] += losses['error_px']
    stats['total_px'] += losses['total_px']

    # log for eval only
    if logger is not None:
        logger.info('Index %d, l1_raw %.4f, rr %.4f, l1 %.4f, occ_be %.4f, epe %.4f, iou %.4f, px error %.4f' %
                    (idx, losses['l1_raw'].item(), losses['rr'].item(), losses['l1'].item(), losses['occ_be'].item(),
                     losses['epe'].item(), losses['iou'].item(), losses['error_px'] / losses['total_px']))

    return outputs, losses, disp
