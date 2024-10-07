import os

from mmengine.config import Config
from mmengine.runner import Runner

# two ways to check early stopping
# 1. early stopping that checks loss -> model "loss" -> parse losses
# 2. early stopping that checks metric -> model "predict" -> val_evaluator ->


def check_early_stopping_runner_loss(folder_path, filename):
    cfg = Config.fromfile(f"{folder_path}/{filename}")
    cfg.work_dir = os.path.join(folder_path, "work_dir_" + filename.split(".")[0])
    runner = Runner.from_cfg(cfg)

    # data = runner.train_dataloader.dataset
    # optim_wrapper = cfg.optim_wrapper
    # runner.model.train_step(data, optim_wrapper)

    # runner.val()

    # output = runner.model(mode="predict")
    # print(output)
    # print(runner.val_evaluator.metrics.metric)
    # print(runner.val_evaluator.metrics.result)

    # runner.run(runner.model, runner.val_data_loader, mode="val")
    # for metric in runner.val_evaluator.metrics:
    #     print(metric)


def check_early_stopping_runner_metric(folder_path, filename):
    cfg = Config.fromfile(f"{folder_path}/{filename}")
    cfg.work_dir = os.path.join(folder_path, "work_dir_" + filename.split(".")[0])
    runner = Runner.from_cfg(cfg)


def check_early_stopping_srun_metric(folder_path, filename):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    # implement metric based early stopping
    cfg.custom_hooks = [
        dict(
            type="EarlyStoppingHook",
            rule="greater",
            monitor="coco/bbox_mAP",  # mAP did not work -> check pascal_voc/mAP' (did get recognized - mmengine - INFO - the monitored metric did not improve in the last 1 records. best score: 0.000. ) and 'coco/bbox_mAP' (did get recognized - the monitored metric did not improve in the last 1 records. best score: 0.000.)
            patience=1,
            min_delta=100000,
        )
    ]
    # convert to iteration based
    cfg.train_cfg = dict(by_epoch=False, max_iters=10000, val_interval=10)
    cfg.default_hooks.logger = dict(type="LoggerHook", log_metric_by_epoch=False)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", by_epoch=False, interval=2000
    )
    cfg.log_processor = dict(by_epoch=False)

    cfg.work_dir = os.path.join(folder_path, "work_dir_" + filename.split(".")[0])

    runner = Runner.from_cfg(cfg)
    runner.train()


def check_early_stopping_srun_loss(folder_path, filename):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    # implement metric based early stopping
    cfg.custom_hooks = [
        dict(
            type="EarlyStoppingHook",
            rule="less",
            monitor="loss",
            patience=1,
            min_delta=100000,
        )
    ]

    # convert to iteration based
    cfg.train_cfg = dict(by_epoch=False, max_iters=10000, val_interval=10)
    cfg.default_hooks.logger = dict(type="LoggerHook", log_metric_by_epoch=False)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", by_epoch=False, interval=2000
    )
    cfg.log_processor = dict(by_epoch=False)

    cfg.work_dir = os.path.join(folder_path, "work_dir_" + filename.split(".")[0])

    runner = Runner.from_cfg(cfg)
    runner.train()


# ?  Skip early stopping process since the evaluation results (dict_keys(['pascal_voc/mAP', 'pascal_voc/AP25', 'pascal_voc/AP30', 'pascal_voc/AP40', 'pascal_voc/AP50', 'pascal_voc/AP70', 'pascal_voc/AP75'])) do not include `monitor` (mAP).
# ? Skip early stopping process since the evaluation results (dict_keys(['pascal_voc/mAP', 'pascal_voc/AP25', 'pascal_voc/AP30', 'pascal_voc/AP40', 'pascal_voc/AP50', 'pascal_voc/AP70', 'pascal_voc/AP75'])) do not include `monitor` (loss).
test_file_0 = "atss_convnext-b_voc0712.py"

# ? Skip early stopping process since the evaluation results (dict_keys(['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l'])) do not include `monitor` (mAP).

test_file_1 = "atss_convnext-b_coco.py"

test_folder_path = "cfg_experiments"

# check_early_stopping_srun_metric(test_folder_path, test_file_0)
check_early_stopping_srun_metric(test_folder_path, test_file_1)
# check_early_stopping_srun_loss(test_folder_path, test_file_0)

# python -m pudb check_early_stopping.py
