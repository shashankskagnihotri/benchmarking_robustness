auto_scale_lr = dict(base_batch_size=50)
backend_args = None
checkpoint_config = dict(interval=0)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 210
model = dict(
    backbone=dict(
        downsample_times=5,
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_stacks=2,
        stage_blocks=[
            2,
            2,
            2,
            2,
            2,
            4,
        ],
        stage_channels=[
            256,
            256,
            384,
            384,
            384,
            512,
        ],
        type='HourglassNet'),
    bbox_head=dict(
        corner_emb_channels=1,
        in_channels=256,
        loss_embedding=dict(
            pull_weight=0.1, push_weight=0.1, type='AssociativeEmbeddingLoss'),
        loss_heatmap=dict(
            alpha=2.0, gamma=4.0, loss_weight=1, type='GaussianFocalLoss'),
        loss_offset=dict(beta=1.0, loss_weight=1, type='SmoothL1Loss'),
        num_classes=80,
        num_feat_levels=2,
        type='CornerHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=None,
    test_cfg=dict(
        corner_topk=100,
        distance_threshold=0.5,
        local_maximum_kernel=3,
        max_per_img=100,
        nms=dict(iou_threshold=0.5, method='gaussian', type='soft_nms'),
        score_thr=0.05),
    train_cfg=None,
    type='CornerNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0005, type='Adam'),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            180,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(
                border=None,
                crop_size=None,
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                ratios=None,
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                test_mode=True,
                test_pad_mode=[
                    'logical_or',
                    127,
                ],
                to_rgb=True,
                type='RandomCenterCropPad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'border',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(
        border=None,
        crop_size=None,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        ratios=None,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_mode=True,
        test_pad_mode=[
            'logical_or',
            127,
        ],
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'border',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=210, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=5,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.5,
                    1.5,
                ),
                hue_delta=18,
                saturation_range=(
                    0.5,
                    1.5,
                ),
                type='PhotoMetricDistortion'),
            dict(
                crop_size=(
                    511,
                    511,
                ),
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                ratios=(
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                ),
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                test_mode=False,
                test_pad_mode=None,
                to_rgb=True,
                type='RandomCenterCropPad'),
            dict(keep_ratio=False, scale=(
                511,
                511,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='PhotoMetricDistortion'),
    dict(
        crop_size=(
            511,
            511,
        ),
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        ratios=(
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_mode=False,
        test_pad_mode=None,
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(keep_ratio=False, scale=(
        511,
        511,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(
        max_per_img=100,
        nms=dict(iou_threshold=0.5, method='gaussian', type='soft_nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    border=None,
                    crop_size=None,
                    mean=[
                        123.675,
                        116.28,
                        103.53,
                    ],
                    ratios=None,
                    std=[
                        58.395,
                        57.12,
                        57.375,
                    ],
                    test_mode=True,
                    test_pad_mode=[
                        'logical_or',
                        127,
                    ],
                    to_rgb=True,
                    type='RandomCenterCropPad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'flip',
                        'flip_direction',
                        'border',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(
                border=None,
                crop_size=None,
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                ratios=None,
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                test_mode=True,
                test_pad_mode=[
                    'logical_or',
                    127,
                ],
                to_rgb=True,
                type='RandomCenterCropPad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'border',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
