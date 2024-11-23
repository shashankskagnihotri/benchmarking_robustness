_base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
crop_size = (1024, 1024)
model = dict(
    data_preprocessor=dict(size=crop_size, corruption=None),
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))