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


coco_metric_cfg = "cfg_experiments/coco_metric_atss_r50_voc0712.py"
naming_new_voc_metric_cfg = "cfg_experiments/voc_metric_atss_r50_voc0712.py"


new_voc_metric_cfg = Config.fromfile(coco_metric_cfg)

new_voc_metric_cfg.data_root = voc0712_data_root
new_voc_metric_cfg.dataset_type = voc0712_dataset_type

new_voc_metric_cfg.train_pipeline = voc0712_train_pipeline
new_voc_metric_cfg.test_pipeline = voc0712_test_pipeline

new_voc_metric_cfg.train_dataloader = voc0712_train_dataloader
new_voc_metric_cfg.val_dataloader = voc0712_val_dataloader
new_voc_metric_cfg.test_dataloader = voc0712_val_dataloader

new_voc_metric_cfg.val_evaluator = voc0712_val_evaluator
new_voc_metric_cfg.test_evaluator = voc0712_val_evaluator


new_voc_metric_cfg.dump(naming_new_voc_metric_cfg)
