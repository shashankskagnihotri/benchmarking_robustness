import os
from mmengine.config import Config
from mmengine.runner import Runner
from rich.traceback import install

install()


paths = ["./configs/", "./projects/"]
model_conf_folders = [
    "fast_rcnn",
    "faster_rcnn",
    "rpn",
    "ssd",
    "retinanet",
    "cascade_rcnn",
    "yolo",
    "cornernet",
    "grid_rcnn",
    "guided_anchoring",
    "fsaf",
    "centernet",
    "libra_rcnn",
    "tridentnet",
    "fcos",
    "reppoints",
    "free_anchor",
    "cascade_rpn",
    "foveabox",
    "double_heads",
    "atss",
    "nas_fcos",
    "centripetalnet",
    "autoassign",
    "sabl",
    "dynamic_rcnn",
    "detr",
    "paa",
    "vfnet",
    "sparse_rcnn",
    "yolof",
    "yolox",
    "deformable_detr",
    "tood",
    "ddod",
    "rtmdet",
    "conditional_detr",
    "dab_detr",
    "dino",
    "glip",
    "ddq",
]
model_proj_folders = [
    "DiffusionDet/configs",
    "EfficientDet/configs",
    "Detic_new/configs",
    "CO-DETR/configs/codino",
]

model_conf_folders = [paths[0] + folder for folder in model_conf_folders]
model_proj_folders = [paths[1] + folder for folder in model_proj_folders]

combined_folders = model_conf_folders + model_proj_folders


# for searching
old_swin_b_backbones = ["swin_b", "swin-b"]
old_swin_l_backbones = ["swin-l", "swinl", "swin_l"]
old_convnext_b_backbones = ["convnext_b"]
old_resnet_50_backbones = ["r50"]
old_resnet_101_backbones = ["r101"]

# ? not used
old_convnext_t_backbones = []

old_backbone_names = [
    old_swin_b_backbones,
    old_swin_l_backbones,
    old_convnext_b_backbones,
    old_convnext_t_backbones,
    old_resnet_50_backbones,
    old_resnet_101_backbones,
]

# main_old_backbone_names = []
# for oldb in old_backbone_names:
#     for suboldb in oldb:
#         main_old_backbone_names.append(suboldb)
# print(main_old_backbone_names)

old_coco_dataset = ["coco"]
old_lvis_dataset = ["lvis"]

old_dataset_names = [old_coco_dataset, old_lvis_dataset]

# main_old_dataset_names = []
# for oldd in old_dataset_names:
#     for suboldd in oldd:
#         main_old_dataset_names.append(suboldd)
# print(main_old_dataset_names)

# print(main_old_backbone_names)

# for naming
# new_backbones = ["swin-b", "swin-l", "convnext-b", "convnext-t", "r50", "r101"]


# ? files_to_use_to_change = []
# for testing just those that only have one implementation

#! add more files
files_to_use_to_change = [
    "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    # DO NOT USE "./mmdetection/configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py",
    "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/ddod/ddod_r50_fpn_1x_coco.py",
    "./mmdetection/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
    "./mmdetection/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py",
]


#! Add SWIN-L -> Ask supervisor
# for specifying the parameters
new_backbone_configs = {
    "swin-b": {
        "type": "SwinTransformer",
        "pretrain_img_size": 384,
        "embed_dims": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.3,
        "patch_norm": True,
        "out_indices": (1, 2, 3),
        "with_cp": True,
        "convert_weights": True,
        "init_cfg": dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
        ),
    },
    "convnext-b": {
        "_delete_": True,
        "type": "mmpretrain.ConvNeXt",
        "arch": "base",
        "out_indices": [1, 2, 3],
        "drop_path_rate": 0.7,
        "layer_scale_init_value": 1.0,
        "gap_before_final_norm": False,
        "with_cp": True,
        "init_cfg": dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_in1k-384px_20221219-4570f792.pth",
            prefix="backbone.",
        ),
    },
    "r101": {
        "type": "ResNet",
        "depth": 101,
        "num_stages": 4,
        "out_indices": [0, 1, 2, 3],
        "frozen_stages": 1,
        "norm_cfg": dict(type="BN", requires_grad=True),
        "norm_eval": True,
        "style": "pytorch",
        "init_cfg": dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    },
    "r50": {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": [0, 1, 2, 3],
        "frozen_stages": 1,
        "norm_cfg": dict(type="BN", requires_grad=True),
        "norm_eval": True,
        "style": "pytorch",
        "init_cfg": dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    },
}


#! Add SWIN-L -> Ask supervisor

new_neck_configs = {
    "swin-b": {"in_channels": [256, 512, 1024]},
    "convnext-b": {"in_channels": [256, 512, 1024]},
    "r101": {"in_channels": [256, 512, 1024, 2048]},
    "r50": {"in_channels": [256, 512, 1024, 2048]},
}


new_dataset_type = {
    "coco": "CocoDataset",
    "lvis": "LVISV1Dataset",
}

new_val_dataloader = {
    "coco": dict(
        batch_size=1,
        dataset=dict(
            ann_file="annotations/instances_val2017.json",
            backend_args=None,
            data_prefix=dict(img="val2017/"),
            data_root="data/coco/",
            pipeline=[
                dict(backend_args=None, type="LoadImageFromFile"),
                dict(
                    keep_ratio=True,
                    scale=(
                        1333,
                        800,
                    ),
                    type="Resize",
                ),
                dict(type="LoadAnnotations", with_bbox=True),
                dict(
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                    ),
                    type="PackDetInputs",
                ),
            ],
            test_mode=True,
            type="CocoDataset",
        ),
        drop_last=False,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(shuffle=False, type="DefaultSampler"),
    ),
    "lvis": dict(
        dataset=dict(
            data_root="data/coco/",
            type="LVISV1Dataset",
            ann_file="annotations/lvis_od_val.json",
            data_prefix=dict(img=""),
        )
    ),
}

new_val_evaluator = {
    "coco": dict(
        ann_file="data/coco/annotations/instances_val2017.json",
        backend_args=None,
        format_only=False,
        metric="bbox",
        type="CocoMetric",
    ),
    "lvis": dict(
        _delete_=True,
        type="LVISFixedAPMetric",
        ann_file="data/coco/" + "annotations/lvis_od_val.json",
    ),
}


necks_we_want = ["double_heads", "dynamic_rcnn"]
backbones_we_want = ["swin-b", "convnext-b", "r50"]
datasets_we_want = ["coco", "lvis"]


def which(path):
    def which_backbone(path):
        backbones = {
            "swin-b": ["swin_b", "swin-b"],
            "swin-l": ["swin-l", "swinl", "swin_l"],
            "convnext-b": ["convnext_b"],
            "r50": ["r50"],
            "r101": ["r101"],
        }

        for backbone, names in backbones.items():
            for name in names:
                if name in path:
                    return backbone

    def which_neck(path):
        return path.split("/")[-2]

    def which_dataset(path):
        if "coco" in path:
            return "coco"
        elif "lvis" in path:
            return "lvis"
        else:
            None

    return which_backbone(path), which_neck(path), which_dataset(path)


for file in files_to_use_to_change:
    backbone, neck, dataset = which(file)

    if backbone and neck and dataset:
        path = "./configs_to_test"
    else:
        path = "./configs_to_train"

    # print(backbone, neck, dataset, file)

    cfg = Config.fromfile(file)
    cfg.checkpoint_config = dict(interval=0)
    destination_file = os.path.join(path, f"{neck}_{backbone}_{dataset}.py")
    cfg.dump(destination_file)


# file = "./configs_to_test/ddod_r50_coco.py"
# cfg = Config.fromfile(file)

# backbone, neck, dataset = which(file)
# cfg.model.backbone = new_backbone_configs["r101"]
# cfg.model.neck.in_channels = new_neck_configs["r101"]["in_channels"]

# cfg.dump("nice try.py")


all_combis = {}
for n in necks_we_want:
    for b in backbones_we_want:
        for d in datasets_we_want:
            all_combis[(n, b, d)] = False

for file in files_to_use_to_change:
    backbone, neck, dataset = which(file)
    if backbone and neck and dataset:
        all_combis[(neck, backbone, dataset)] = True

print(all_combis)


#! Take in more files
reference_configs = {
    "double_heads": "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    "dynamic_rcnn": "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
}

for (neck, backbone, dataset), found in all_combis.items():
    if not found:
        reference_file = reference_configs[neck]
        backbone_ref, neck_ref, dataset_ref = which(reference_file)
        cfg = Config.fromfile(reference_file)

        assert (
            neck == neck_ref
        ), f"Neck mismatch: {neck} != {neck_ref}, make neck is in refercence list"

        if backbone != backbone_ref:
            cfg.model.backbone = new_backbone_configs[backbone]
            cfg.model.neck.in_channels = new_neck_configs[backbone]["in_channels"]

        if dataset != dataset_ref:
            cfg.dataset_type = new_dataset_type[dataset]
            cfg.val_dataloader = new_val_dataloader[dataset]
            cfg.val_evaluator = new_val_evaluator[dataset]
            cfg.test_dataloader = new_val_dataloader[dataset]
            cfg.test_evaluator = new_val_evaluator[dataset]

        destination_file = os.path.join(
            "./configs_to_train", f"{neck}_{backbone}_{dataset}.py"
        )
        cfg.dump(destination_file)
        all_combis[(neck, backbone, dataset)] = True


print(all_combis)
print(
    f"{sum(all_combis.values())} files created, {len(all_combis) - sum(all_combis.values())} missing"
)

#! only for testing
# for configs_train in os.listdir("./configs_to_train"):
#     print(configs_train)
#     cfg = Config.fromfile(f"./configs_to_train/{configs_train}")
#     cfg.work_dir = "./work_dirs/"
#     cfg.total_epochs = 1
#     runner = Runner.from_cfg(cfg)
#     runner.train()


#! implement dataset
# https://github.com/open-mmlab/mmdetection/blob/main/configs/glip/lvis/glip_atss_swin-t_a_fpn_dyhead_pretrain_zeroshot_lvis.py
# dataset_type = 'LVISV1Dataset'
# data_root = 'data/coco/'

# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         type=dataset_type,
#         ann_file='annotations/lvis_od_val.json',
#         data_prefix=dict(img='')))
# test_dataloader = val_dataloader

# # numpy < 1.24.0
# val_evaluator = dict(
#     _delete_=True,
#     type='LVISFixedAPMetric',
#     ann_file=data_root + 'annotations/lvis_od_val.json')
# test_evaluator = val_evaluator
