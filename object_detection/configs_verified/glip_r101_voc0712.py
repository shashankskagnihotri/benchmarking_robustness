auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
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
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        style='pytorch',
        type='ResNet'),
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
        num_classes=80,
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
    clip_grad=None,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=2e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
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
test_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
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
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
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
        times=3,
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
val_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
