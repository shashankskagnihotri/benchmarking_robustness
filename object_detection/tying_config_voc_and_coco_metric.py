import os
from mmengine.config import Config
from mmdetection.configs._base_.datasets.voc0712 import (
    data_root as voc0712_data_root,
    dataset_type as voc0712_dataset_type,
    train_pipeline as voc0712_train_pipeline,
    test_pipeline as voc0712_test_pipeline,
    train_dataloader as voc0712_train_dataloader,
    val_dataloader as voc0712_val_dataloader,
)

reference_file = "./voc_coco_metric_testfiles/atss_r50_coco_voc_val_test.py"
save_under_file = "./voc_coco_metric_testfiles/atss_r50_voc_cocometric_json_train_txt_val_txt_test_txt.py"

cfg = Config.fromfile(reference_file)


cfg.data_root = voc0712_data_root
cfg.dataset_type = voc0712_dataset_type

cfg.train_pipeline = voc0712_train_pipeline
cfg.test_pipeline = voc0712_test_pipeline

cfg.train_dataloader = voc0712_train_dataloader
cfg.val_dataloader = voc0712_val_dataloader
cfg.test_dataloader = voc0712_val_dataloader


cfg.val_evaluator = (
    dict(
        ann_file="data/VOCdevkit/voc_coco_format/voc0712_val.json",
        backend_args=None,
        format_only=False,
        metric="bbox",
        type="CocoMetric",
    ),
)
cfg.test_evaluator = dict(
    ann_file="data/VOCdevkit/voc_coco_format/voc07_test.json",
    backend_args=None,
    format_only=False,
    metric="bbox",
    type="CocoMetric",
)

cfg.dump(save_under_file)
