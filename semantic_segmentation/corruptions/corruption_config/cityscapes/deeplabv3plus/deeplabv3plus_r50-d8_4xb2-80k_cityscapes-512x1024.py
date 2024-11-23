_base_ = [
    '../../../../mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../../../mmsegmentation/configs/_base_/datasets/cityscapes.py', '../../../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../../../mmsegmentation/configs/_base_/schedules/schedule_80k.py'
]

crop_size = (512, 1024)
model = dict(
    data_preprocessor=dict(size=crop_size,
                         corruption=None))