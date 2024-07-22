from mmengine.config import Config
from mmdetection.configs._base_.datasets.voc0712 import (
    data_root as voc0712_data_root,
    dataset_type as voc0712_dataset_type,
    train_pipeline as voc0712_train_pipeline,
    test_pipeline as voc0712_test_pipeline,
    train_dataloader as voc0712_train_dataloader,
    val_dataloader as voc0712_val_dataloader,
    val_evaluator as voc0712_val_evaluator,
)

param_scheduler = [
    dict(begin=0, by_epoch=False, end=4000, start_factor=0.00025, type="LinearLR"),
    dict(
        begin=0,
        by_epoch=True,
        end=25,
        gamma=0.1,
        milestones=[
            22,
            24,
        ],
        type="MultiStepLR",
    ),
]
train_cfg = dict(max_epochs=25, type="EpochBasedTrainLoop", val_interval=5)


coco_fmt_vocmetric = "configs_verified/centernet_r50_voc0712_vocmetric.py"


naming_coco_fmt_vocmetric_3x_p_t = (
    "configs_verified/centernet_r50_voc0712_vocmetric_3x_p_t.py"
)
naming_coco_fmt_cocometric_3x_p_t = (
    "configs_verified/centernet_r50_voc0712_cocometric_3x_p_t.py"
)

naming_voc0712_metric = "configs_verified/centernet_r50_voc0712_vocmetric_0712.py"


voc0712_metric_cfg = Config.fromfile(coco_fmt_vocmetric)

voc0712_metric_cfg.data_root = voc0712_data_root
voc0712_metric_cfg.dataset_type = voc0712_dataset_type

voc0712_metric_cfg.train_pipeline = voc0712_train_pipeline
voc0712_metric_cfg.test_pipeline = voc0712_test_pipeline

voc0712_metric_cfg.train_dataloader = voc0712_train_dataloader
voc0712_metric_cfg.val_dataloader = voc0712_val_dataloader
voc0712_metric_cfg.test_dataloader = voc0712_val_dataloader

voc0712_metric_cfg.val_evaluator = voc0712_val_evaluator
voc0712_metric_cfg.test_evaluator = voc0712_val_evaluator


voc0712_metric_cfg.dump(naming_voc0712_metric)


cfg_coco_fmt_vocmetric_3x_p_t = Config.fromfile(coco_fmt_vocmetric)
cfg_coco_fmt_cocometric_3x_p_t = Config.fromfile(coco_fmt_vocmetric)


cfg_coco_fmt_vocmetric_3x_p_t.train_cfg = train_cfg
cfg_coco_fmt_vocmetric_3x_p_t.param_scheduler = param_scheduler

cfg_coco_fmt_cocometric_3x_p_t.train_cfg = train_cfg
cfg_coco_fmt_cocometric_3x_p_t.param_scheduler = param_scheduler


cfg_coco_fmt_vocmetric_3x_p_t.dump(naming_coco_fmt_vocmetric_3x_p_t)
cfg_coco_fmt_cocometric_3x_p_t.dump(naming_coco_fmt_cocometric_3x_p_t)
