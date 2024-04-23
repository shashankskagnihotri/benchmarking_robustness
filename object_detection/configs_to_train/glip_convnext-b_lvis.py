auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'data/coco/'
dataset_type = 'LVISV1Dataset'
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
        _delete_=True,
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
    dataset=dict(
        ann_file='annotations/lvis_od_val.json',
        data_prefix=dict(img=''),
        data_root='data/coco/',
        type='LVISV1Dataset'))
test_evaluator = dict(
    _delete_=True,
    ann_file='data/coco/annotations/lvis_od_val.json',
    type='LVISFixedAPMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        backend='pillow',
        keep_ratio=True,
        scale=(
            800,
            1333,
        ),
        type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train2017.json',
            backend_args=None,
            data_prefix=dict(img='train2017/'),
            data_root='data/coco/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(
                    backend_args=None,
                    imdecode_backend='pillow',
                    type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='GTBoxSubOne_GLIP'),
                dict(
                    backend='pillow',
                    keep_ratio=True,
                    resize_type='FixScaleResize',
                    scales=[
                        (
                            1333,
                            480,
                        ),
                        (
                            1333,
                            560,
                        ),
                        (
                            1333,
                            640,
                        ),
                        (
                            1333,
                            720,
                        ),
                        (
                            1333,
                            800,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(prob=0.5, type='RandomFlip_GLIP'),
                dict(min_gt_bbox_wh=(
                    1,
                    1,
                ), type='FilterAnnotations'),
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                        'text',
                        'custom_entities',
                    ),
                    type='PackDetInputs'),
            ],
            return_classes=True,
            type='CocoDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GTBoxSubOne_GLIP'),
    dict(
        backend='pillow',
        keep_ratio=True,
        resize_type='FixScaleResize',
        scales=[
            (
                1333,
                480,
            ),
            (
                1333,
                560,
            ),
            (
                1333,
                640,
            ),
            (
                1333,
                720,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip_GLIP'),
    dict(min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'custom_entities',
        ),
        type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/lvis_od_val.json',
        data_prefix=dict(img=''),
        data_root='data/coco/',
        type='LVISV1Dataset'))
val_evaluator = dict(
    _delete_=True,
    ann_file='data/coco/annotations/lvis_od_val.json',
    type='LVISFixedAPMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
