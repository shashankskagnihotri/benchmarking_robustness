num_classes = 20
#! assign num_classes = 20 at right places in the config files
#! if epochbased -> max epochs = max epochs / 3

METAINFO = {
    "classes": (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ),
    # palette is a list of color tuples, which is used for visualization.
    "palette": [
        (106, 0, 228),
        (119, 11, 32),
        (165, 42, 42),
        (0, 0, 192),
        (197, 226, 255),
        (0, 60, 100),
        (0, 0, 142),
        (255, 77, 255),
        (153, 69, 1),
        (120, 166, 157),
        (0, 182, 199),
        (0, 226, 252),
        (182, 182, 255),
        (0, 0, 230),
        (220, 20, 60),
        (163, 255, 0),
        (0, 82, 0),
        (3, 95, 161),
        (0, 80, 100),
        (183, 130, 88),
    ],
}

# dataset settings
dataset_type = "CocoDataset"
data_root = "data/VOCdevkit/"


#! FileNotFoundError: [Errno 2] No such file or directory: voc_coco_fmt_annotations/voc07_test.json
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1000, 600), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_dataloader = dict(
    dataset=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            # _delete_=True,
            type=dataset_type,
            data_root=data_root,
            ann_file="voc_coco_fmt_annotations/voc0712_trainval.json", # changed from annotations/....
            data_prefix=dict(img=""),
            metainfo=METAINFO,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=None,
        ),
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root, #! didn´t find the config
        ann_file="voc_coco_fmt_annotations/voc07_test.json", # changed from annotations/....
        data_prefix=dict(img=""),
        metainfo=METAINFO,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "voc_coco_fmt_annotations/voc07_test.json", # changed from annotations/....
    metric="bbox",
    format_only=False,
    backend_args=None,
)
test_evaluator = val_evaluator

# training schedule, the dataset is repeated 3 times, so the
# actual epoch = 4 * 3 = 12
max_epochs = 4
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
