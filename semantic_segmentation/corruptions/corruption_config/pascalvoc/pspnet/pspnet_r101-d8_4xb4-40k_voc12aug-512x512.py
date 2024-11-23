_base_ = './pspnet_r50-d8_4xb4-40k_voc12aug-512x512.py'
crop_size = (512, 512)
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    data_preprocessor=dict(
        size=crop_size,
        corruption = None),
    backbone=dict(depth=101),
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))