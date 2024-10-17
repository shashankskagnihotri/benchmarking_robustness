_base_ = [
    '../../mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py', '../../mmsegmentation/configs/_base_/datasets/ade20k.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py', '../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head= dict(num_classes=150),
    auxiliary_head = dict(num_classes=150),
    test_cfg=dict(crop_size=(512, 512), stride=(42, 42)))
train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader


# crop_size = (64, 64)
# data_preprocessor = dict(size=crop_size)
# model = dict(
#     data_preprocessor=data_preprocessor,
#     test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
# train_dataloader = dict(batch_size=1, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader
# The original problem comes from the validation set. You also need to specifiy that it has the correct size


# TOO LOW MIOU: 06/02 02:18:17 - mmengine - INFO - Iter(val) [2000/2000]    aAcc: 30.1600  mIoU: 1.2900  mAcc: 2.3500  data_time: 0.0013  time: 0.1520


