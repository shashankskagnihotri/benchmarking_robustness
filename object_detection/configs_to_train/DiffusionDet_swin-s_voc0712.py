auto_scale_lr = dict(base_batch_size=16, enable=True)
backend = 'pillow'
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
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.DiffusionDet.diffusiondet',
    ])
data_root = 'data/VOCdevkit/'
dataset_type = 'VOCDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=False,
        interval=75000,
        max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=False, type='LogProcessor', window_size=50)
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
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    bbox_head=dict(
        criterion=dict(
            assigner=dict(
                candidate_topk=5,
                center_radius=2.5,
                match_costs=[
                    dict(
                        alpha=0.25,
                        eps=1e-08,
                        gamma=2.0,
                        type='FocalLossCost',
                        weight=2.0),
                    dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='DiffusionDetMatcher'),
            loss_bbox=dict(loss_weight=5.0, reduction='sum', type='L1Loss'),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=2.0,
                reduction='sum',
                type='FocalLoss',
                use_sigmoid=True),
            loss_giou=dict(loss_weight=2.0, reduction='sum', type='GIoULoss'),
            num_classes=20,
            type='DiffusionDetCriterion'),
        ddim_sampling_eta=1.0,
        deep_supervision=True,
        feat_channels=256,
        num_classes=20,
        num_heads=6,
        num_proposals=500,
        prior_prob=0.01,
        roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        sampling_timesteps=1,
        single_head=dict(
            act_cfg=dict(inplace=True, type='ReLU'),
            dim_feedforward=2048,
            dropout=0.0,
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
            num_cls_convs=1,
            num_heads=8,
            num_reg_convs=3,
            type='SingleDiffusionDetHead'),
        snr_scale=2.0,
        type='DynamicDiffusionDetHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    test_cfg=dict(
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        score_thr=0.5,
        use_nms=True),
    type='DiffusionDet')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
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
test_cfg = dict(_scope_='mmdet', type='TestLoop')
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
val_cfg = dict(_scope_='mmdet', type='ValLoop')
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
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
