METAINFO = dict(
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
    ])
auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
data_root = 'data/VOCdevkit/'
dataset_type = 'CocoDataset'
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
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
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
            2048,
        ],
        num_outs=5,
        out_channels=256,
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
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
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
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
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
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
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
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
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
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
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
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[
            16,
            19,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/voc07_test.json',
        data_prefix=dict(img=''),
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
train_cfg = dict(max_epochs=6, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            _delete_=True,
            ann_file='annotations/voc0712_trainval.json',
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
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/voc07_test.json',
        data_prefix=dict(img=''),
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
