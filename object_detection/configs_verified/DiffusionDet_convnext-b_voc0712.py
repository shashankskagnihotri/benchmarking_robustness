auto_scale_lr = dict(base_batch_size=16, enable=True)
backend = 'pillow'
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.DiffusionDet.diffusiondet',
    ])
data_root = 'data/VOCdevkit/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=False,
        interval=75000,
        max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
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
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.7,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_in1k-384px_20221219-4570f792.pth',
            prefix='backbone.',
            type='Pretrained'),
        layer_scale_init_value=1.0,
        out_indices=[
            1,
            2,
            3,
        ],
        type='mmpretrain.ConvNeXt',
        with_cp=True),
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
            256,
            512,
            1024,
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
    _scope_='mmdet',
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=2.5e-05, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.01, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=450000,
        gamma=0.1,
        milestones=[
            350000,
            420000,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='voc_coco_fmt_annotations/voc07_test.json',
        data_prefix=dict(img=''),
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
        type='CocoDataset'))
test_evaluator = dict(
    ann_file='data/VOCdevkit/voc_coco_fmt_annotations/voc07_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
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
    max_iters=450000, type='IterBasedTrainLoop', val_interval=75000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            ann_file='voc_coco_fmt_annotations/voc0712_trainval.json',
            backend_args=None,
            data_prefix=dict(img=''),
            data_root='data/VOCdevkit/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
                dict(keep_ratio=True, scale=(
                    1000,
                    600,
                ), type='Resize'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=3,
        type='RepeatDataset'))
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
        ann_file='voc_coco_fmt_annotations/voc07_test.json',
        data_prefix=dict(img=''),
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
        type='CocoDataset'))
val_evaluator = dict(
    ann_file='data/VOCdevkit/voc_coco_fmt_annotations/voc07_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
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
