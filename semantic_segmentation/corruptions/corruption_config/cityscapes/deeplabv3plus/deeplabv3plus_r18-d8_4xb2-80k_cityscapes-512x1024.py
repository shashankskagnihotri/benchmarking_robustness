_base_ = './deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
crop_size = (512, 1024)
model = dict(
    data_preprocessor=dict(size=crop_size,
                         corruption=None),
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))