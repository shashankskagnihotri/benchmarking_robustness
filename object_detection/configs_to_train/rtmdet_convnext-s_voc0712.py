auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
base_lr = 0.001
checkpoint = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=90,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    1280,
                    1280,
                ),
                type='RandomResize'),
            dict(crop_size=(
                1280,
                1280,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    1280,
                    1280,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
    dict(monitor='pascal_voc/mAP', type='EarlyStoppingHook'),
]
data_root = 'data/VOCdevkit/'
dataset_type = 'VOCDataset'
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
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
img_scales = [
    (
        1280,
        1280,
    ),
    (
        640,
        640,
    ),
    (
        1920,
        1920,
    ),
]
interval = 10
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 36
model = dict(
    backbone=dict(
        arch='small',
        drop_path_rate=0.6,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth',
            prefix='backbone.',
            type='Pretrained'),
        layer_scale_init_value=1.0,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        type='mmpretrain.ConvNeXt'),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
                64,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=True,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(num_groups=32, type='GN'),
        num_classes=20,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
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
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        norm_cfg=dict(num_groups=32, type='GN'),
        num_csp_blocks=3,
        out_channels=256,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
norm_cfg = dict(num_groups=32, type='GN')
optim_wrapper = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0002, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.7, decay_type='layer_wise', num_layers=12),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            27,
            33,
        ],
        type='MultiStepLR'),
]
resume = False
stage2_num_epochs = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='VOC2007/ImageSets/Main/test.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007/'),
        data_root='data/VOCdevkit/',
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            ),
            palette=[
                (
                    106,
                    0,
                    228,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    197,
                    226,
                    255,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    153,
                    69,
                    1,
                ),
                (
                    120,
                    166,
                    157,
                ),
                (
                    0,
                    182,
                    199,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    163,
                    255,
                    0,
                ),
                (
                    0,
                    82,
                    0,
                ),
                (
                    3,
                    95,
                    161,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    183,
                    130,
                    88,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VOCDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    eval_mode='11points',
    iou_thrs=[
        0.25,
        0.3,
        0.4,
        0.5,
        0.7,
        0.75,
    ],
    metric='mAP',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='VOCMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1000,
        600,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            90,
            1,
        ),
    ],
    max_epochs=36,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    backend_args=None,
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    data_root='data/VOCdevkit/',
                    filter_cfg=dict(
                        bbox_min_size=32, filter_empty_gt=True, min_size=32),
                    metainfo=dict(
                        classes=(
                            'aeroplane',
                            'bicycle',
                            'bird',
                            'boat',
                            'bottle',
                            'bus',
                            'car',
                            'cat',
                            'chair',
                            'cow',
                            'diningtable',
                            'dog',
                            'horse',
                            'motorbike',
                            'person',
                            'pottedplant',
                            'sheep',
                            'sofa',
                            'train',
                            'tvmonitor',
                        ),
                        palette=[
                            (
                                106,
                                0,
                                228,
                            ),
                            (
                                119,
                                11,
                                32,
                            ),
                            (
                                165,
                                42,
                                42,
                            ),
                            (
                                0,
                                0,
                                192,
                            ),
                            (
                                197,
                                226,
                                255,
                            ),
                            (
                                0,
                                60,
                                100,
                            ),
                            (
                                0,
                                0,
                                142,
                            ),
                            (
                                255,
                                77,
                                255,
                            ),
                            (
                                153,
                                69,
                                1,
                            ),
                            (
                                120,
                                166,
                                157,
                            ),
                            (
                                0,
                                182,
                                199,
                            ),
                            (
                                0,
                                226,
                                252,
                            ),
                            (
                                182,
                                182,
                                255,
                            ),
                            (
                                0,
                                0,
                                230,
                            ),
                            (
                                220,
                                20,
                                60,
                            ),
                            (
                                163,
                                255,
                                0,
                            ),
                            (
                                0,
                                82,
                                0,
                            ),
                            (
                                3,
                                95,
                                161,
                            ),
                            (
                                0,
                                80,
                                100,
                            ),
                            (
                                183,
                                130,
                                88,
                            ),
                        ]),
                    pipeline=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            keep_ratio=True,
                            scale=(
                                1000,
                                600,
                            ),
                            type='Resize'),
                        dict(prob=0.5, type='RandomFlip'),
                        dict(type='PackDetInputs'),
                    ],
                    type='VOCDataset'),
                dict(
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    backend_args=None,
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    data_root='data/VOCdevkit/',
                    filter_cfg=dict(
                        bbox_min_size=32, filter_empty_gt=True, min_size=32),
                    metainfo=dict(
                        classes=(
                            'aeroplane',
                            'bicycle',
                            'bird',
                            'boat',
                            'bottle',
                            'bus',
                            'car',
                            'cat',
                            'chair',
                            'cow',
                            'diningtable',
                            'dog',
                            'horse',
                            'motorbike',
                            'person',
                            'pottedplant',
                            'sheep',
                            'sofa',
                            'train',
                            'tvmonitor',
                        ),
                        palette=[
                            (
                                106,
                                0,
                                228,
                            ),
                            (
                                119,
                                11,
                                32,
                            ),
                            (
                                165,
                                42,
                                42,
                            ),
                            (
                                0,
                                0,
                                192,
                            ),
                            (
                                197,
                                226,
                                255,
                            ),
                            (
                                0,
                                60,
                                100,
                            ),
                            (
                                0,
                                0,
                                142,
                            ),
                            (
                                255,
                                77,
                                255,
                            ),
                            (
                                153,
                                69,
                                1,
                            ),
                            (
                                120,
                                166,
                                157,
                            ),
                            (
                                0,
                                182,
                                199,
                            ),
                            (
                                0,
                                226,
                                252,
                            ),
                            (
                                182,
                                182,
                                255,
                            ),
                            (
                                0,
                                0,
                                230,
                            ),
                            (
                                220,
                                20,
                                60,
                            ),
                            (
                                163,
                                255,
                                0,
                            ),
                            (
                                0,
                                82,
                                0,
                            ),
                            (
                                3,
                                95,
                                161,
                            ),
                            (
                                0,
                                80,
                                100,
                            ),
                            (
                                183,
                                130,
                                88,
                            ),
                        ]),
                    pipeline=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            keep_ratio=True,
                            scale=(
                                1000,
                                600,
                            ),
                            type='Resize'),
                        dict(prob=0.5, type='RandomFlip'),
                        dict(type='PackDetInputs'),
                    ],
                    type='VOCDataset'),
            ],
            ignore_keys=[
                'dataset_type',
            ],
            type='ConcatDataset'),
        times=1,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1000,
        600,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1280,
            1280,
        ),
        type='RandomResize'),
    dict(crop_size=(
        1280,
        1280,
    ), type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        1280,
        1280,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1280,
                    1280,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    1920,
                    1920,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        1920,
                        1920,
                    ),
                    type='Pad'),
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
                        'scale_factor',
                        'flip',
                        'flip_direction',
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
        ann_file='VOC2007/ImageSets/Main/test.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007/'),
        data_root='data/VOCdevkit/',
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            ),
            palette=[
                (
                    106,
                    0,
                    228,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    165,
                    42,
                    42,
                ),
                (
                    0,
                    0,
                    192,
                ),
                (
                    197,
                    226,
                    255,
                ),
                (
                    0,
                    60,
                    100,
                ),
                (
                    0,
                    0,
                    142,
                ),
                (
                    255,
                    77,
                    255,
                ),
                (
                    153,
                    69,
                    1,
                ),
                (
                    120,
                    166,
                    157,
                ),
                (
                    0,
                    182,
                    199,
                ),
                (
                    0,
                    226,
                    252,
                ),
                (
                    182,
                    182,
                    255,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    220,
                    20,
                    60,
                ),
                (
                    163,
                    255,
                    0,
                ),
                (
                    0,
                    82,
                    0,
                ),
                (
                    3,
                    95,
                    161,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    183,
                    130,
                    88,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1000,
                600,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VOCDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    eval_mode='11points',
    iou_thrs=[
        0.25,
        0.3,
        0.4,
        0.5,
        0.7,
        0.75,
    ],
    metric='mAP',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='VOCMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
