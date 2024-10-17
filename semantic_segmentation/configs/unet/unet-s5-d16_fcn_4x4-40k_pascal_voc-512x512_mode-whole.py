_base_ = [
    '../../mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py', '../../mmsegmentation/configs/_base_/datasets/pascal_voc12.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py', '../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head= dict(num_classes=21),
    auxiliary_head = dict(num_classes=21),
    test_cfg=dict(mode='whole'))
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# ERROR: AssertionError: The input image size (512, 699) should be divisible by the whole downsample rate 16, when num_stages is 5, strides is (1, 1, 1, 1, 1), and downsamples is (True, True, True, True).


