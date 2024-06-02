checkpoint_config = dict(
    by_epoch=False, create_symlink=False, interval=4000, max_keep_ckpts=10)
crop_size = (
    896,
    896,
)
data = dict(
    samples_per_gpu=1,
    test=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            flip=False,
            img_scale=(
                3584,
                896,
            ),
            transforms=[
                dict(keep_ratio=True, type='Resize'),
                dict(size_divisor=32, type='ResizeToMultiple'),
                dict(type='RandomFlip'),
                dict(
                    mean=[
                        122.771,
                        116.746,
                        104.094,
                    ],
                    std=[
                        68.5,
                        66.632,
                        70.323,
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
            3584,
            896,
        ), ratio_range=(
            0.5,
            2.0,
        ), type='Resize'),
        dict(cat_max_ratio=0.75, crop_size=(
            896,
            896,
        ), type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PhotoMetricDistortion'),
        dict(
            mean=[
                122.771,
                116.746,
                104.094,
            ],
            std=[
                68.5,
                66.632,
                70.323,
            ],
            to_rgb=True,
            type='Normalize'),
        dict(pad_val=0, seg_pad_val=255, size=(
            896,
            896,
        ), type='Pad'),
        dict(type='ToMask'),
        dict(type='DefaultFormatBundle'),
        dict(
            keys=[
                'img',
                'gt_semantic_seg',
                'gt_masks',
                'gt_labels',
            ],
            type='Collect'),
    ]),
    val=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            flip=False,
            img_scale=(
                3584,
                896,
            ),
            transforms=[
                dict(keep_ratio=True, type='Resize'),
                dict(size_divisor=32, type='ResizeToMultiple'),
                dict(type='RandomFlip'),
                dict(
                    mean=[
                        122.771,
                        116.746,
                        104.094,
                    ],
                    std=[
                        68.5,
                        66.632,
                        70.323,
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
data_root = 'data/ade/ADEChallengeData2016'
dataset_type = 'ADE20KDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
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
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
img_norm_cfg = dict(
    mean=[
        122.771,
        116.746,
        104.094,
    ],
    std=[
        68.5,
        66.632,
        70.323,
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
load_from = '../checkpoint_files/onepeace_seg_cocostuff2ade20k.pth'
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
    backbone=dict(
        attention_heads=24,
        bucket_size=56,
        cffn_ratio=0.25,
        conv_inplane=64,
        deform_num_heads=24,
        deform_ratio=0.5,
        drop_path_rate=0.5,
        dropout=0.0,
        embed_dim=1536,
        ffn_embed_dim=6144,
        init_values=1e-06,
        interaction_indexes=[
            [
                0,
                9,
            ],
            [
                10,
                19,
            ],
            [
                20,
                29,
            ],
            [
                30,
                39,
            ],
        ],
        layers=40,
        n_points=4,
        rp_bias=True,
        shared_rp_bias=False,
        type='OnePeaceAdapter',
        use_checkpoint=True,
        with_cp=True),
    decode_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=1024,
        in_channels=[
            1536,
            1536,
            1536,
            1536,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=150,
        num_queries=200,
        num_stuff_classes=50,
        num_things_classes=100,
        num_transformer_feat_level=3,
        out_channels=1024,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        batch_first=False,
                        dropout=0.0,
                        embed_dims=1024,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=32,
                        num_levels=3,
                        num_points=4,
                        type='MultiScaleDeformableAttention'),
                    ffn_cfgs=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=1024,
                        feedforward_channels=4096,
                        ffn_drop=0.0,
                        num_fcs=2,
                        type='FFN',
                        with_cp=True),
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder'),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(
                normalize=True, num_feats=512, type='SinePositionalEncoding'),
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(
            normalize=True, num_feats=512, type='SinePositionalEncoding'),
        transformer_decoder=dict(
            init_cfg=None,
            num_layers=9,
            return_intermediate=True,
            transformerlayers=dict(
                attn_cfgs=dict(
                    attn_drop=0.0,
                    batch_first=False,
                    dropout_layer=None,
                    embed_dims=1024,
                    num_heads=32,
                    proj_drop=0.0,
                    type='MultiheadAttention'),
                feedforward_channels=4096,
                ffn_cfgs=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=1024,
                    feedforward_channels=4096,
                    ffn_drop=0.0,
                    num_fcs=2,
                    with_cp=True),
                operation_order=(
                    'cross_attn',
                    'norm',
                    'self_attn',
                    'norm',
                    'ffn',
                    'norm',
                ),
                type='DetrTransformerDecoderLayer'),
            type='DetrTransformerDecoder'),
        type='Mask2FormerHead'),
    init_cfg=None,
    pretrained=
    '/pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/configs/onepeace/one-peace-vision.pkl',
    test_cfg=dict(
        crop_size=(
            896,
            896,
        ),
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=100,
        mode='slide',
        panoptic_on=True,
        semantic_on=False,
        stride=(
            512,
            512,
        )),
    train_cfg=dict(
        assigner=dict(
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            dice_cost=dict(
                eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
            type='MaskHungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
    type='EncoderDecoderMask2Former')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 150
num_stuff_classes = 50
num_things_classes = 100
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    constructor='OnePeaceLearningRateDecayOptimizerConstructor',
    lr=1e-05,
    paramwise_cfg=dict(decay_rate=0.95, num_layers=40),
    type='AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
pretrained = '/pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/configs/onepeace/one-peace-vision.pkl'
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
            3584,
            896,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(size_divisor=32, type='ResizeToMultiple'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    122.771,
                    116.746,
                    104.094,
                ],
                std=[
                    68.5,
                    66.632,
                    70.323,
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
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
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
        3584,
        896,
    ), ratio_range=(
        0.5,
        2.0,
    ), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(
        896,
        896,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(
        mean=[
            122.771,
            116.746,
            104.094,
        ],
        std=[
            68.5,
            66.632,
            70.323,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(pad_val=0, seg_pad_val=255, size=(
        896,
        896,
    ), type='Pad'),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        keys=[
            'img',
            'gt_semantic_seg',
            'gt_masks',
            'gt_labels',
        ],
        type='Collect'),
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
work_dir = '../work_dirs/mask2former_onepeace_adapter_g_896_40k_ade20k_ss'
