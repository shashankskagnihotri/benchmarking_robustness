# import sys
# sys.path.insert(
#     0,
#     "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/mmdetection/mmdet/datasets/transforms/",
# )
# from mmdetection.mmdet.datasets.transforms.load_voc_captions import LoadCaptions

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import torch

from mmengine.runner import set_random_seed
import sys

sys.path.append("./mmdetection")
sys.path.append("./mmdetection/projects")


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="config file path")
    parser.add_argument("work_dirs", help="work_dirs path")
    args = parser.parse_args()
    return args


def trainer(
    config,
    work_dir=None,
    auto_scale_lr=False,
    amp=False,
    resume=None,
    cfg_options=None,
    launcher="none",
    local_rank=0,
):
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    set_random_seed(seed=0)

    # load config
    cfg = Config.fromfile(config)
    cfg.launcher = launcher
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if work_dir is not None:
        # update configs according to CLI args if work_dir is not None
        cfg.work_dir = work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(config))[0])

    # enable automatic-mixed-precision training
    if amp is True:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    # enable automatically scaling LR
    if auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                'Can not find "auto_scale_lr" or '
                '"auto_scale_lr.enable" or '
                '"auto_scale_lr.base_batch_size" in your'
                " configuration file."
            )

    # resume is determined in this priority: resume from > auto_resume
    if resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif resume is not None:
        cfg.resume = True
        cfg.load_from = resume

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # # # start training
    runner.train()

    # # Mixed precision training setup
    # # start training
    # with torch.cuda.amp.autocast():
    #     runner.train()


if __name__ == "__main__":
    args = parse_args()
    trainer(
        config=args.config,
        work_dir=args.work_dirs,
        auto_scale_lr=True,
        amp=False,
        resume="auto",
        cfg_options=None,
        launcher="none",
        local_rank=0,
    )


# python -m pudb new_trainer.py horeka_test_submission/atss_convnext-b_coco_wandb-gradient-logging.py cfg_experiments/slum_experiments/nan/atss_seeded

# python -m pudb new_trainer.py horeka_test_submission/cascade_rcnn_convnext-b_coco_no-mixed-precision-training.py cfg_experiments/slum_experiments/nan
# job 20240920_094025

# python -m pudb new_trainer.py horeka_test_submission/cascade_rcnn_swin-b_coco_no-mixed-precision-training.py cfg_experiments/slum_experiments/nan
# job


# python -m pudb new_trainer.py horeka_test_submission/train_fully/codino_swin-s_coco.py cfg_experiments/slum_experiments

# python new_trainer.py cfg_experiments/codino_convnext-s_coco.py cfg_experiments/codino_convnext-s_coco

# 24546337


"""
When running srun 

/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1702400366987/work/aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/new_trainer.py", line 111, in <module>
    trainer(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/new_trainer.py", line 101, in trainer
    runner.train()
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/runner/runner.py", line 1777, in train
    model = self.train_loop.run()  # type: ignore
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/runner/loops.py", line 98, in run
    self.run_epoch()
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/runner/loops.py", line 115, in run_epoch
    self.run_iter(idx, data_batch)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/runner/loops.py", line 131, in run_iter
    outputs = self.runner.model.train_step(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/model/base_model/base_model.py", line 114, in train_step
    losses = self._run_forward(data, mode='loss')  # type: ignore
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/model/base_model/base_model.py", line 361, in _run_forward
    results = self(**data, mode=mode)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/mmdetection/mmdet/models/detectors/base.py", line 92, in forward
    return self.loss(inputs, data_samples)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/mmdetection/projects/CO-DETR/codetr/codetr.py", line 171, in loss
    bbox_losses, x = self.query_head.loss(x, batch_data_samples)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/mmdetection/projects/CO-DETR/codetr/co_dino_head.py", line 314, in loss
    outs = self(x, batch_img_metas, dn_label_query, dn_bbox_query,
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/mmdetection/projects/CO-DETR/codetr/co_dino_head.py", line 134, in forward
    self.transformer(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/benchmarking_robustness/object_detection/mmdetection/projects/CO-DETR/codetr/transformer.py", line 1159, in forward
    memory = self.encoder(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmcv/cnn/bricks/transformer.py", line 941, in forward
    query = layer(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/fairscale/nn/checkpoint/checkpoint_activations.py", line 190, in _checkpointed_forward
    output = CheckpointFunction.apply(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/autograd/function.py", line 539, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/fairscale/nn/checkpoint/checkpoint_activations.py", line 282, in forward
    outputs = run_function(*unpacked_args, **unpacked_kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmcv/cnn/bricks/transformer.py", line 830, in forward
    query = self.attentions[attn_index](
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmengine/utils/misc.py", line 395, in new_func
    output = old_func(*args, **kwargs)
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmcv/ops/multi_scale_deform_attn.py", line 369, in forward
    output = MultiScaleDeformableAttnFunction.apply(
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/torch/autograd/function.py", line 539, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/pfs/work7/workspace/scratch/ma_ruweber-team_project_fss2024/miniconda3/envs/benchmark/lib/python3.10/site-packages/mmcv/ops/multi_scale_deform_attn.py", line 64, in forward
    output = ext_module.ms_deform_attn_forward(
RuntimeError: ms_deform_attn_impl_forward: implementation for device cuda:0 not found."""
