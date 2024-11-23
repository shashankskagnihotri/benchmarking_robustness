_base_ = './deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
crop_size = (512, 1024)
model = dict(
    data_preprocessor=dict(size=crop_size,
                         corruption=None),
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101))