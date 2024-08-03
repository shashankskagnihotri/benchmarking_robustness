auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
cls_layer = dict(
    norm_temperature=50.0,
    norm_weight=True,
    type='ZeroShotClassifier',
    use_bias=0.0,
    zs_weight_dim=512,
    zs_weight_path='data/metadata/lvis_v1_clip_a+cname.npy')
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.Detic_new.detic',
    ])
data_root = 'data/VOCdevkit/'
dataset_cls = dict(
    ann_file='annotations/imagenet_lvis_image_info.json',
    backend_args=None,
    data_prefix=dict(img='ImageNet-LVIS/'),
    data_root='data/imagenet',
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=False, with_label=True),
        dict(
            keep_ratio=True,
            ratio_range=(
                0.5,
                1.5,
            ),
            scale=(
                448,
                448,
            ),
            type='RandomResize'),
        dict(
            allow_negative_crop=True,
            bbox_clip_border=False,
            crop_size=(
                448,
                448,
            ),
            crop_type='absolute_range',
            recompute_bbox=False,
            type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PackDetInputs'),
    ],
    type='ImageNetLVISV1Dataset')
dataset_det = dict(
    dataset=dict(
        ann_file='annotations/lvis_v1_train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='data/lvis/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    896,
                    896,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    896,
                    896,
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
        type='LVISV1Dataset'),
    oversample_thr=0.001,
    type='ClassBalancedDataset')
dataset_type = 'VOCDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=False,
        interval=30000,
        max_keep_ckpts=5,
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
find_unused_parameters = True
image_size_cls = (
    448,
    448,
)
image_size_det = (
    896,
    896,
)
load_from = './first_stage/detic_centernet2_swin-b_fpn_4x_lvis_boxsup.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
lvis_cat_frequency_info = 'data/metadata/lvis_v1_train_cat_info.json'
max_iter = 180000
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        mlp_ratio=4,
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        out_indices=(
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
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
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
                cat_freq_path='data/metadata/lvis_v1_train_cat_info.json',
                cls_predictor_cfg=dict(
                    norm_temperature=50.0,
                    norm_weight=True,
                    type='ZeroShotClassifier',
                    use_bias=0.0,
                    zs_weight_dim=512,
                    zs_weight_path='data/metadata/lvis_v1_clip_a+cname.npy'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=0.1, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=20,
                reg_class_agnostic=True,
                reg_predictor_cfg=[
                    dict(in_features=1024, out_features=1024, type='Linear'),
                    dict(inplace=True, type='ReLU'),
                    dict(in_features=1024, out_features=4, type='Linear'),
                ],
                roi_feat_size=7,
                type='DeticBBoxHead',
                use_fed_loss=True),
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
                cat_freq_path='data/metadata/lvis_v1_train_cat_info.json',
                cls_predictor_cfg=dict(
                    norm_temperature=50.0,
                    norm_weight=True,
                    type='ZeroShotClassifier',
                    use_bias=0.0,
                    zs_weight_dim=512,
                    zs_weight_path='data/metadata/lvis_v1_clip_a+cname.npy'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=0.1, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=20,
                reg_class_agnostic=True,
                reg_predictor_cfg=[
                    dict(in_features=1024, out_features=1024, type='Linear'),
                    dict(inplace=True, type='ReLU'),
                    dict(in_features=1024, out_features=4, type='Linear'),
                ],
                roi_feat_size=7,
                type='DeticBBoxHead',
                use_fed_loss=True),
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
                cat_freq_path='data/metadata/lvis_v1_train_cat_info.json',
                cls_predictor_cfg=dict(
                    norm_temperature=50.0,
                    norm_weight=True,
                    type='ZeroShotClassifier',
                    use_bias=0.0,
                    zs_weight_dim=512,
                    zs_weight_path='data/metadata/lvis_v1_clip_a+cname.npy'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=0.1, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=20,
                reg_class_agnostic=True,
                reg_predictor_cfg=[
                    dict(in_features=1024, out_features=1024, type='Linear'),
                    dict(inplace=True, type='ReLU'),
                    dict(in_features=1024, out_features=4, type='Linear'),
                ],
                roi_feat_size=7,
                type='DeticBBoxHead',
                use_fed_loss=True),
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
            num_classes=20,
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
            1.0,
            1.0,
            1.0,
        ],
        type='DeticRoIHead'),
    rpn_head=dict(
        conv_bias=True,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(eps=1e-06, loss_weight=1.0, type='GIoULoss'),
        loss_cls=dict(
            alpha=0.25,
            beta=4.0,
            gamma=2.0,
            ignore_high_fp=0.85,
            loss_weight=1.0,
            neg_weight=0.5,
            pos_weight=0.5,
            type='HeatmapFocalLoss'),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=20,
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
                    add_gt_as_proposals=False,
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
                    add_gt_as_proposals=False,
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
            nms=dict(iou_threshold=0.9, type='nms'),
            nms_pre=4000,
            score_thr=0.0001)),
    type='Detic')
num_classes = 20
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(T_max=180000, begin=0, by_epoch=False, type='CosineAnnealingLR'),
]
reg_layer = [
    dict(in_features=1024, out_features=1024, type='Linear'),
    dict(inplace=True, type='ReLU'),
    dict(in_features=1024, out_features=4, type='Linear'),
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
train_cfg = dict(
    max_iters=180000, type='IterBasedTrainLoop', val_interval=180000)
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
            type='ConcatDataset')),
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
train_pipeline_cls = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            1.5,
        ),
        scale=(
            448,
            448,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        bbox_clip_border=False,
        crop_size=(
            448,
            448,
        ),
        crop_type='absolute_range',
        recompute_bbox=False,
        type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_pipeline_det = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            896,
            896,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            896,
            896,
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
val_pipeline = [
    dict(backend_args=None, imdecode_backend=None, type='LoadImageFromFile'),
    dict(backend=None, keep_ratio=True, scale=(
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
