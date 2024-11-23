crop_size = (
    640,
    640,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
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
    checkpoint=dict(by_epoch=False, interval=32000, type='CheckpointHook'),
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
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = '../checkpoint_files/ade20k/upernet/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        embed_dims=1024,
        img_size=(
            640,
            640,
        ),
        in_channels=3,
        init_values=1e-06,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        norm_eval=False,
        num_heads=16,
        num_layers=24,
        out_indices=[
            7,
            11,
            15,
            23,
        ],
        patch_size=16,
        qv_bias=True,
        type='BEiT'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        corruption='gaussian_noise',
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
        channels=1024,
        dropout_ratio=0.1,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
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
    neck=dict(
        embed_dim=1024, rescales=[
            4,
            2,
            1,
            0.5,
        ], type='Feature2Pyramid'),
    pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    test_cfg=dict(crop_size=(
        640,
        640,
    ), mode='slide', stride=(
        426,
        426,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=2,
    constructor='LayerDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=2e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.95, num_layers=24),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=3000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=3000,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
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
                2560,
                640,
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
    dict(keep_ratio=True, scale=(
        2560,
        640,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=320000, type='IterBasedTrainLoop', val_interval=32000)
train_dataloader = dict(
    batch_size=1,
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
                    2560,
                    640,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    640,
                    640,
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
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2560,
            640,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
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
                2560,
                640,
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
work_dir = '../corruptions/work_dirs/ade20k/upernet/beit-large_upernet_8xb1-amp-160k_ade20k-640x640/gaussian_noise'
