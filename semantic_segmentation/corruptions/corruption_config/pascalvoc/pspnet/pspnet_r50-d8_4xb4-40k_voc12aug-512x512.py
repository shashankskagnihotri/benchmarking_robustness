_base_ = [
    '../../../../mmsegmentation/configs/_base_/models/pspnet_r50-d8.py',
    '../../../../mmsegmentation/configs/_base_/datasets/pascal_voc12_aug.py', '../../../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
model = dict(
    data_preprocessor=dict(
        size=crop_size,
        corruption = None),
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))