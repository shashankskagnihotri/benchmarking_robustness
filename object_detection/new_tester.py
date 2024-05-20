# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("work_dirs", help="work_dirs path")
    args = parser.parse_args()
    return args


def tester(
    config_path,
    checkpoint_path,
    work_dir=None,
    out_file=None,
    show=False,
    show_dir=None,
    wait_time=2,
    cfg_options=None,
    launcher="none",
    tta=False,
    local_rank=0,
):
    # Set the LOCAL_RANK environment variable
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Reduce the number of repeated compilations and improve testing speed
    setup_cache_size_limit_of_dynamo()

    # Load config
    cfg = Config.fromfile(config_path)
    cfg.launcher = launcher
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    # Determine work_dir priority: function argument > config file > default
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(config_path))[0]
        )

    cfg.load_from = checkpoint_path

    if show or show_dir:
        cfg = trigger_visualization_hook(cfg, show, show_dir, wait_time)

    if tta:
        if "tta_model" not in cfg:
            warnings.warn(
                "Cannot find ``tta_model`` in config, we will set it as default."
            )
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            )
        if "tta_pipeline" not in cfg:
            warnings.warn(
                "Cannot find ``tta_pipeline`` in config, we will set it as default."
            )
            test_data_cfg = cfg.test_dataloader.dataset
            while "dataset" in test_data_cfg:
                test_data_cfg = test_data_cfg["dataset"]
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [
                        dict(
                            type="PackDetInputs",
                            meta_keys=(
                                "img_id",
                                "img_path",
                                "ori_shape",
                                "img_shape",
                                "scale_factor",
                                "flip",
                                "flip_direction",
                            ),
                        )
                    ],
                ],
            )
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # Build the runner from config
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Add `DumpResults` dummy metric if out_file is specified
    if out_file is not None:
        assert out_file.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=out_file))

    # Start testing
    runner.test()


if __name__ == "__main__":
    args = parse_args()
    tester(
        args.config,
        args.checkpoint,
        args.work_dir,
        out=None,
        show=False,
        show_dir=None,
        wait_time=2,
        cfg_options=None,
        launcher="none",
        tta=False,
        local_rank=0,
    )
