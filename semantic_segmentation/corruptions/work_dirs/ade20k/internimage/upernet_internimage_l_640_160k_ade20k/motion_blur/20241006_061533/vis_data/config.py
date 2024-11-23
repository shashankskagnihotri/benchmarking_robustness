checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
crop_size = (
    640,
    640,
)
data = dict(
    samples_per_gpu=2,
    test=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            flip=False,
            img_scale=(
                2560,
                640,
            ),
            transforms=[
                dict(keep_ratio=True, type='Resize'),
                dict(size_divisor=32, type='ResizeToMultiple'),
                dict(type='RandomFlip'),
                dict(
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
                    to_rgb=True,
                    type='Normalize'),
                dict(keys=[
                    'img',
                ], type='ImageToTensor'),
                dict(keys=[
                    'img',
                ], type='Collect'),
            ],
            type='MultiScaleFlipAug'),
    ]),
    train=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(reduce_zero_label=True, type='LoadAnnotations'),
        dict(img_scale=(
            2560,
            640,
        ), ratio_range=(
            0.5,
            2.0,
        ), type='Resize'),
        dict(cat_max_ratio=0.75, crop_size=(
            640,
            640,
        ), type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PhotoMetricDistortion'),
        dict(
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
            to_rgb=True,
            type='Normalize'),
        dict(pad_val=0, seg_pad_val=255, size=(
            640,
            640,
        ), type='Pad'),
        dict(type='DefaultFormatBundle'),
        dict(keys=[
            'img',
            'gt_semantic_seg',
        ], type='Collect'),
    ]),
    val=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            flip=False,
            img_scale=(
                2560,
                640,
            ),
            transforms=[
                dict(keep_ratio=True, type='Resize'),
                dict(size_divisor=32, type='ResizeToMultiple'),
                dict(type='RandomFlip'),
                dict(
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
                    to_rgb=True,
                    type='Normalize'),
                dict(keys=[
                    'img',
                ], type='ImageToTensor'),
                dict(keys=[
                    'img',
                ], type='Collect'),
            ],
            type='MultiScaleFlipAug'),
    ]))
data_preprocessor = dict(
    bgr_to_rgb=True,
    corruption=None,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/ade/ADEChallengeData2016'
dataset_type = 'ADE20KDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
img_norm_cfg = dict(
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
    to_rgb=True)
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = '../checkpoint_files/ade20k/internimage/upernet_internimage_l_640_160k_ade20k.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr_config = dict(
    _delete_=True,
    by_epoch=False,
    min_lr=0.0,
    policy='poly',
    power=1.0,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=640,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        channels=160,
        core_op='DCNv3',
        depths=[
            5,
            5,
            22,
            5,
        ],
        drop_path_rate=0.4,
        groups=[
            10,
            20,
            40,
            80,
        ],
        init_cfg=dict(
            checkpoint=
            'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth',
            type='Pretrained'),
        layer_scale=1.0,
        mlp_ratio=4.0,
        norm_layer='LN',
        offset_scale=2.0,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        post_norm=True,
        type='InternImage',
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        corruption='motion_blur',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            640,
            640,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            160,
            320,
            640,
            1280,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    constructor='CustomLayerDecayOptimizerConstructor',
    lr=2e-05,
    paramwise_cfg=dict(
        depths=[
            5,
            5,
            22,
            5,
        ],
        layer_decay_rate=0.94,
        num_layers=37,
        offset_lr_scale=1.0),
    type='AdamW',
    weight_decay=0.05)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth'
resume = False
runner = dict(type='IterBasedRunner')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='data/ade/ADEChallengeData2016',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            2560,
            640,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(size_divisor=32, type='ResizeToMultiple'),
            dict(type='RandomFlip'),
            dict(
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
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        data_root='data/ade/ADEChallengeData2016',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(img_scale=(
        2560,
        640,
    ), ratio_range=(
        0.5,
        2.0,
    ), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(
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
        to_rgb=True,
        type='Normalize'),
    dict(pad_val=0, seg_pad_val=255, size=(
        640,
        640,
    ), type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_semantic_seg',
    ], type='Collect'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='data/ade/ADEChallengeData2016',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '../corruptions/work_dirs/ade20k/internimage/upernet_internimage_l_640_160k_ade20k/motion_blur'
