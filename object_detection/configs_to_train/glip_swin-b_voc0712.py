auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(monitor='pascal_voc/mAP', type='EarlyStoppingHook'),
]
data_root = 'data/VOCdevkit/'
dataset_type = 'VOCDataset'
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
lang_model_name = 'bert-base-uncased'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 100
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
            1,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            4,
            8,
            16,
            32,
            64,
        ],
        out_indices=[
            1,
            2,
            3,
            4,
        ],
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        strides=[
            4,
            2,
            2,
            2,
            2,
        ],
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoderForGLIP'),
        feat_channels=256,
        in_channels=256,
        lang_model_name='bert-base-uncased',
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=20,
        type='ATSSVLFusionHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    language_model=dict(name='bert-base-uncased', type='BertModel'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='FPN_DropBlock'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            iou_calculator=dict(type='BboxOverlaps2D_GLIP'),
            topk=9,
            type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='GLIP')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=50,
        begin=50,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=5e-05,
        type='CosineAnnealingLR'),
]
resume = False
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
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
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
