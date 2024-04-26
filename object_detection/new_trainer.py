# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def train(
    config,
    cfg_options=None,
    work_dir=None,
    amp=False,
    auto_scale_lr=None,
    resume="auto",  #! Originally NONE but we want it to resume training if not done -> "auto"
    launcher="none",
):
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(config)
    cfg.launcher = launcher
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
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

    # start training
    runner.train()
