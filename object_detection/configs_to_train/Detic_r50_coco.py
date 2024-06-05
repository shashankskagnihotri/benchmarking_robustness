auto_scale_lr = dict(base_batch_size=64, enable=True)
backend = 'pillow'
backend_args = None
batch_augments = [
    dict(size=(
        1024,
        1024,
    ), type='BatchFixedSizePad'),
]
cls_layer = dict(
    norm_temperature=50.0,
    norm_weight=True,
    type='ZeroShotClassifier',
    use_bias=0.0,
    zs_weight_dim=512,
    zs_weight_path='rand')
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.Detic.detic',
    ])
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet', interval=1, max_keep_ckpts=2, type='CheckpointHook'),
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
image_size = (
    1024,
    1024,
)
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 25
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
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
    data_preprocessor=dict(
        batch_augments=[
            dict(size=(
                1024,
                1024,
            ), type='BatchFixedSizePad'),
        ],
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
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        init_cfg=dict(layer='Conv2d', type='Caffe2Xavier'),
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
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
                    type='DeltaXYWHBBoxCoder'),
                cls_predictor_cfg=dict(
                    norm_temperature=50.0,
                    norm_weight=True,
                    type='ZeroShotClassifier',
                    use_bias=0.0,
                    zs_weight_dim=512,
                    zs_weight_path='rand'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=80,
                reg_class_agnostic=True,
                reg_predictor_cfg=[
                    dict(in_features=1024, out_features=1024, type='Linear'),
                    dict(inplace=True, type='ReLU'),
                    dict(in_features=1024, out_features=4, type='Linear'),
                ],
                roi_feat_size=7,
                type='DeticBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                cls_predictor_cfg=dict(
                    norm_temperature=50.0,
                    norm_weight=True,
                    type='ZeroShotClassifier',
                    use_bias=0.0,
                    zs_weight_dim=512,
                    zs_weight_path='rand'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=80,
                reg_class_agnostic=True,
                reg_predictor_cfg=[
                    dict(in_features=1024, out_features=1024, type='Linear'),
                    dict(inplace=True, type='ReLU'),
                    dict(in_features=1024, out_features=4, type='Linear'),
                ],
                roi_feat_size=7,
                type='DeticBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                cls_predictor_cfg=dict(
                    norm_temperature=50.0,
                    norm_weight=True,
                    type='ZeroShotClassifier',
                    use_bias=0.0,
                    zs_weight_dim=512,
                    zs_weight_path='rand'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=80,
                reg_class_agnostic=True,
                reg_predictor_cfg=[
                    dict(in_features=1024, out_features=1024, type='Linear'),
                    dict(inplace=True, type='ReLU'),
                    dict(in_features=1024, out_features=4, type='Linear'),
                ],
                roi_feat_size=7,
                type='DeticBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                8,
                16,
                32,
            ],
            finest_scale=112,
            out_channels=256,
            roi_layer=dict(
                output_size=7,
                sampling_ratio=0,
                type='RoIAlign',
                use_torchvision=True),
            type='SingleRoIExtractor'),
        mask_head=dict(
            class_agnostic=True,
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=80,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                8,
                16,
                32,
            ],
            finest_scale=112,
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='DeticRoIHead'),
    rpn_head=dict(
        conv_bias=True,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            neg_weight=0.75,
            pos_weight=0.25,
            type='GaussianFocalLoss'),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=80,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='CenterNetRPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=300,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.02),
        rpn=dict(
            max_per_img=256,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.9, type='nms'),
            nms_pre=1000,
            score_thr=0.0001)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.8,
                    neg_iou_thr=0.8,
                    pos_iou_thr=0.8,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='CascadeRCNN')
num_classes = 80
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.04, momentum=0.9, type='SGD', weight_decay=4e-05),
    paramwise_cfg=dict(norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=4000,
        start_factor=0.00025,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=25,
        gamma=0.1,
        milestones=[
            22,
            24,
        ],
        type='MultiStepLR'),
]
reg_layer = [
    dict(in_features=1024, out_features=1024, type='Linear'),
    dict(inplace=True, type='ReLU'),
    dict(in_features=1024, out_features=4, type='Linear'),
]
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='Resize'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
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
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        backend='pillow', keep_ratio=True, scale=(
            1333,
            800,
        ), type='Resize'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
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
    _scope_='mmdet', max_epochs=25, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        _scope_='mmdet',
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
                    keep_ratio=True,
                    ratio_range=(
                        0.1,
                        2.0,
                    ),
                    scale=(
                        1024,
                        1024,
                    ),
                    type='RandomResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        1024,
                        1024,
                    ),
                    crop_type='absolute_range',
                    recompute_bbox=True,
                    type='RandomCrop'),
                dict(min_gt_bbox_wh=(
                    0.01,
                    0.01,
                ), type='FilterAnnotations'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=8,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmdet',
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(
        _scope_='mmdet',
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type='absolute_range',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(
        _scope_='mmdet',
        min_gt_bbox_wh=(
            0.01,
            0.01,
        ),
        type='FilterAnnotations'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='Resize'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
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
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco/annotations/instances_val2017.json',
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
