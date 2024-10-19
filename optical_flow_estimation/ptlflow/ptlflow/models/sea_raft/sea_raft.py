from argparse import ArgumentParser, Namespace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .corr import get_corr_block
from .utils import coords_grid
from .extractor import ResNetFPN
from .layer import conv3x3
from ..base_model.base_model import BaseModel

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""
        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)

        flow_loss = 0.0
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            loss_i = outputs["nf_preds"][i]
            final_mask = (
                (~torch.isnan(loss_i.detach()))
                & (~torch.isinf(loss_i.detach()))
                & valid
            )
            flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

        return flow_loss


class SEARAFT(BaseModel):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=8)

        self.output_dim = args.dim * 2

        self.args.corr_channel = args.corr_levels * (args.corr_radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(
            args,
            input_dim=6,
            output_dim=2 * self.args.dim,
            norm_layer=nn.BatchNorm2d,
            init_weight=True,
        )

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0),
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1),
        )
        if args.iters > 0:
            self.fnet = ResNetFPN(
                args,
                input_dim=3,
                output_dim=self.output_dim,
                norm_layer=nn.BatchNorm2d,
                init_weight=True,
            )
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

        if self.args.alternate_corr and alt_cuda_corr is None:
            print(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--corr_levels", type=int, default=4)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument("--dim", type=int, default=128)
        parser.add_argument("--initial_dim", type=int, default=64)
        parser.add_argument("--num_blocks", type=int, default=2)
        parser.add_argument("--block_dims", type=int, nargs="+", default=[64, 128, 256])
        parser.add_argument(
            "--pretrain", type=str, default="resnet18", choices=("resnet18, resnet34")
        )
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=1000.0)
        parser.add_argument("--iters", type=int, default=4)
        parser.add_argument("--alternate_corr", action="store_true")
        parser.add_argument("--not_use_var", action="store_false", dest="use_var")
        parser.add_argument("--var_min", type=float, default=0)
        parser.add_argument("--var_max", type=float, default=10)
        return parser

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)
        coords2 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        if "flows" in inputs:
            flow_gt = inputs["flows"][:, 0]
        else:
            N, _, H, W = image1.shape
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        flow_predictions = []
        info_predictions = []

        # run the context network
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.args.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = get_corr_block(
                fmap1=fmap1_8x,
                fmap2=fmap2_8x,
                radius=self.args.corr_radius,
                num_levels=self.args.corr_levels,
                alternate_corr=self.args.alternate_corr,
            )

        for itr in range(self.args.iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (
                coords_grid(N, H, W, dtype=image1.dtype, device=image1.device) + flow_8x
            ).detach()
            corr = corr_fn(coords2)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            info_up = self.postprocess_predictions(
                info_up, image_resizer, is_flow=False
            )
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if self.training:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min

                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                # Large b Component
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                # Small b Component
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                # term2: [N, 2, m, H, W]
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (
                    torch.exp(-log_b).unsqueeze(1)
                )
                # term1: [N, m, H, W]
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(
                    weight, dim=1, keepdim=True
                ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

        if self.training:
            outputs = {
                "flows": flow_up[:, None],
                "flow_preds": flow_predictions,
                "info_preds": info_predictions,
                "nf_preds": nf_predictions,
            }
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": flow_8x}

        return outputs


class SEARAFT_S(SEARAFT):
    pretrained_checkpoints = {
        "tartan": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_s-tartan-f7e26f21.ckpt",
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_s-chairs-6980249f.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_s-things-a15c1713.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_s-sintel-bb63371a.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_s-kitti-3a96c1cc.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_s-spring-4d13c106.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        args.iters = 4
        args.pretrain = "resnet18"
        super().__init__(args)


class SEARAFT_M(SEARAFT):
    pretrained_checkpoints = {
        "tartan": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-tartan-e684ed5f.ckpt",
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-chairs-1cb7b11e.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-things-ac45dd7f.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-sintel-f8bb7e3f.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-kitti-e51f7603.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-spring-de7c13e2.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        args.iters = 4
        args.pretrain = "resnet34"
        super().__init__(args)


class SEARAFT_L(SEARAFT):
    pretrained_checkpoints = {
        "tartan": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-tartan-e684ed5f.ckpt",
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-chairs-1cb7b11e.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-things-ac45dd7f.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-sintel-f8bb7e3f.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-kitti-e51f7603.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-spring-de7c13e2.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        args.iters = 12
        args.pretrain = "resnet34"
        super().__init__(args)
