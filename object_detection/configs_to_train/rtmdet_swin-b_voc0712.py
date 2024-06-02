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
]
data_root = 'data/VOCdevkit/'
dataset_type = 'CocoDataset'
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
max_epochs = 33
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
        out_indices=(
            1,
            2,
            3,
            4,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        strides=(
            4,
            2,
            2,
            2,
            2,
        ),
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
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
            256,
            512,
            1024,
            2048,
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
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=333, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=50,
        begin=16,
        by_epoch=True,
        convert_to_iter_based=True,
        end=33,
        eta_min=5e-05,
        type='CosineAnnealingLR'),
]
resume = False
stage2_num_epochs = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=5,
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
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
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
    dynamic_intervals=[
        (
            90,
            1,
        ),
    ],
    max_epochs=33,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_dataloader = dict(
    batch_size=16,
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
    batch_size=5,
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
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
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
