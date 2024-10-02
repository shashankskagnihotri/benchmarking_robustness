_base_ = './deeplabv3_r50-d8_4xb4-160k_ade20k-512x512.py'
crop_size = (512, 512)
model = dict(data_preprocessor=dict(size=crop_size, corruption=None),
            pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))