import os
from mmengine.config import Config
# from mmdetection.configs._base_.datasets.voc0712 import (
#     backend_args as backend_args,
# )


from voc0712_cocofmt_reference import (
    METAINFO as voc0712_METAINFO,
    data_root as voc0712_data_root,
    dataset_type as voc0712_dataset_type,
    train_pipeline as voc0712_train_pipeline,
    test_pipeline as voc0712_test_pipeline,
    train_dataloader as voc0712_train_dataloader,
    val_dataloader as voc0712_val_dataloader,
)

#! how to import num classes for each voc model in the right way !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from rich.traceback import install

install()


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
files_which_we_have = [
    "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    # DO NOT USE "./mmdetection/configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py",
    "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/ddod/ddod_r50_fpn_1x_coco.py",
    "./mmdetection/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
    "./mmdetection/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py",
    #! added more but not sure if right
    "./mmdetection/configs/fast_rcnn/fast-rcnn_r50_fpn_2x_coco.py",  #! not pretrained might have to train
    "./mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py",  #! not pretrained might have to train
    "./mmdetection/configs/rpn/rpn_r50_fpn_2x_coco.py",
    "./mmdetection/configs/retinanet/retinanet_r50_fpn_ms-640-800-3x_coco.py",
    "./mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_20e_coco.py",
    "./mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_ms-3x_coco.py",
    "./mmdetection/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py",
    "./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py",
    "./mmdetection/configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/reppoints/reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py",
    "./mmdetection/configs/free_anchor/freeanchor_r50_fpn_1x_coco.py",
    "./mmdetection/configs/foveabox/fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
    "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/atss/atss_r50_fpn_1x_coco.py",
    "./mmdetection/configs/sabl/sabl-cascade-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/detr/detr_r50_8xb2-150e_coco.py",
    "./mmdetection/configs/paa/paa_r50_fpn_ms-3x_coco.py",
    "./mmdetection/configs/vfnet/vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco.py",
    "./mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
    "./mmdetection/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py",
    "./mmdetection/configs/tood/tood_r50_fpn_ms-2x_coco.py",
    "./mmdetection/configs/ddod/ddod_r50_fpn_1x_coco.py",
    "./mmdetection/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
    "./mmdetection/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py",
    "./mmdetection/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py",
    "./mmdetection/configs/ddq/ddq-detr-5scale_r50_8xb2-12e_coco.py",
    "./mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py",
    "./mmdetection/configs/fast_rcnn/fast-rcnn_r101_fpn_2x_coco.py",
    "./mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_ms-3x_coco.py",
    "./mmdetection/configs/rpn/rpn_r101_fpn_2x_coco.py",
    "./mmdetection/configs/retinanet/retinanet_r101_fpn_ms-640-800-3x_coco.py",
    "./mmdetection/configs/cascade_rcnn/cascade-rcnn_r101_fpn_20e_coco.py",
    "./mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_r101_fpn_ms-3x_coco.py",
    "./mmdetection/configs/grid_rcnn/grid-rcnn_r101_fpn_gn-head_2x_coco.py",
    "./mmdetection/configs/fsaf/fsaf_r101_fpn_1x_coco.py",
    "./mmdetection/configs/libra_rcnn/libra-faster-rcnn_r101_fpn_1x_coco.py",
    "./mmdetection/configs/reppoints/reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py",
    "./mmdetection/configs/free_anchor/freeanchor_r101_fpn_1x_coco.py",
    "./mmdetection/configs/foveabox/fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
    "./mmdetection/configs/atss/atss_r101_fpn_1x_coco.py",
    "./mmdetection/configs/sabl/sabl-cascade-rcnn_r101_fpn_1x_coco.py",
    "./mmdetection/configs/paa/paa_r101_fpn_ms-3x_coco.py",
    "./mmdetection/configs/vfnet/vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco.py",
    "./mmdetection/configs/sparse_rcnn/sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
    "./mmdetection/configs/tood/tood_r101_fpn_ms-2x_coco.py",
    "./mmdetection/configs/centernet/centernet-update_r50_fpn_8xb8-amp-lsj-200e_coco.py",
    # "./mmdetection/configs/fcos/fcos_r50_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py",
    "./mmdetection/configs/centernet/centernet-update_r101_fpn_8xb8-amp-lsj-200e_coco.py",
    # "./mmdetection/configs/fcos/fcos_r101_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py",
    "./mmdetection/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py",
    #! should add swin-b and convnext files such that they do not get retrained unnecessarily
    "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py",
    #! add swin-b
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_4xb32-100e_coco.py",
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py",
    "./mmdetection/projects/Detic/configs/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k.py",
    #! others
    "./mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
    # "./mmdetection/configs/cornernet/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco.py",
    "./mmdetection/configs/guided_anchoring/ga-faster-rcnn_x101-64x4d_fpn_1x_coco.py",
    # "./mmdetection/configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py",
    "./mmdetection/configs/yolox/yolox_x_8xb8-300e_coco.py",
    "./mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
    "./mmdetection/projects/EfficientDet/configs/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco.py",
    # "./mmdetection/projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py",
    #! add swin-l when everything is done
]


#! Add SWIN-L -> Ask supervisor
# for specifying the parameters
new_backbone_configs = {
    "swin-b": {
        "type": "SwinTransformer",
        "pretrain_img_size": 384,
        "embed_dims": 128,
        "depths": [2, 2, 18, 2, 1],
        "num_heads": [4, 8, 16, 32, 64],
        "strides": [4, 2, 2, 2, 2],
        "window_size": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.3,
        "patch_norm": True,
        "out_indices": [1, 2, 3, 4],
        "with_cp": True,
        "convert_weights": True,
        "init_cfg": dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
        ),
    },
    "convnext-b": {
        # "_delete_": True, # gave errors when tried to train
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
            prefix="backbone.",  # ? maybe get rid of prefix
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
    "swin-b": {"in_channels": [256, 512, 1024, 2048]},
    "convnext-b": {"in_channels": [256, 512, 1024]},
    "r101": {"in_channels": [256, 512, 1024, 2048]},
    "r50": {"in_channels": [256, 512, 1024, 2048]},
}


# new_dataset_type = {
#     "coco": "CocoDataset",
#     "voc0712": "VOCDataset",
#     # "lvis": "LVISV1Dataset",
# }

#! seperatly set "data_root": "data/coco/" ???
#! came till here with checking
# new_train_dataloader_dataset = {
#     "voc0712": dict(
#         dataset=dict(
#             ignore_keys=["dataset_type"],
#             datasets=[
#                 dict(
#                     type="VOCDataset",
#                     data_root="data/VOCdevkit/",
#                     ann_file="VOC2007/ImageSets/Main/trainval.txt",
#                     data_prefix=dict(sub_data_root="VOC2007/"),
#                     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#                     pipeline=train_pipeline,
#                 ),
#                 dict(
#                     type="VOCDataset",
#                     data_root="data/VOCdevkit/",
#                     ann_file="VOC2012/ImageSets/Main/trainval.txt",
#                     data_prefix=dict(sub_data_root="VOC2012/"),
#                     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#                     pipeline=train_pipeline,
#                 ),
#             ],
#         )
#     ),
# }


# new_val_dataloader = {
#     "coco": dict(
#         batch_size=1,
#         dataset=dict(
#             ann_file="annotations/instances_val2017.json",
#             backend_args=None,
#             data_prefix=dict(img="val2017/"),
#             data_root="data/coco/",
#             pipeline=[
#                 dict(backend_args=None, type="LoadImageFromFile"),
#                 dict(
#                     keep_ratio=True,
#                     scale=(
#                         1333,
#                         800,
#                     ),
#                     type="Resize",
#                 ),
#                 dict(type="LoadAnnotations", with_bbox=True),
#                 dict(
#                     meta_keys=(
#                         "img_id",
#                         "img_path",
#                         "ori_shape",
#                         "img_shape",
#                         "scale_factor",
#                     ),
#                     type="PackDetInputs",
#                 ),
#             ],
#             test_mode=True,
#             type="CocoDataset",
#         ),
#         drop_last=False,
#         num_workers=2,
#         persistent_workers=True,
#         sampler=dict(shuffle=False, type="DefaultSampler"),
#     ),
# "voc0712": dict(),
# "lvis": dict(
#     dataset=dict(
#         data_root="data/coco/",
#         type="LVISV1Dataset",
#         ann_file="annotations/lvis_od_val.json",
#         data_prefix=dict(img=""),
#     )
# ),
# }

# new_val_evaluator = {
#     "coco": dict(
#         ann_file="data/coco/annotations/instances_val2017.json",
#         backend_args=None,
#         format_only=False,
#         metric="bbox",
#         type="CocoMetric",
#     ),
#     "voc0712": dict(),
#     # "lvis": dict(
#     #     _delete_=True,
#     #     type="LVISFixedAPMetric",
#     #     ann_file="data/coco/" + "annotations/lvis_od_val.json",
#     # ),
# }

#! extend to all
# necks_we_want = ["double_heads", "dynamic_rcnn", ]
necks_we_want = [
    # ? got to go through till yolo
    "fast_rcnn",
    "faster_rcnn",
    "rpn",
    # "ssd",  #! only caffe implemetation
    "retinanet",
    "cascade_rcnn",
    "yolo",
    # "cornernet",  #! has neck=none
    "grid_rcnn",
    "guided_anchoring",
    "fsaf",
    "centernet",
    "libra_rcnn",
    # "tridentnet",  # ! only caffe implemetation
    # "fcos",  #! only caffe or resnext implemetation
    "reppoints",
    "free_anchor",
    # "cascade_rpn", #! only caffe implemetation
    "foveabox",
    "double_heads",
    "atss",
    # "nas_fcos", #! only caffe implemetation
    # "centripetalnet", #! has none neck
    # "autoassign",  #! only caffe implemetation
    "sabl",
    "dynamic_rcnn",
    "detr",
    "paa",
    "vfnet",
    "sparse_rcnn",
    # "yolof",  #! only caffe implemetation
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
    "Detic",
    "EfficientDet",
    "DiffusionDet",
    "codino",
    # "vitdet", #! too complicated
]
backbones_we_want = [
    "swin-b",
    "convnext-b",
    "r50",
    "r101",
]  #! extend for swin-l when ready
datasets_we_want = [
    "coco",
    "voc0712",
    # "lvis"
]


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
        neck = path.split("/")[-2]
        if neck == "configs":
            neck = path.split("/")[-3]
        return neck

    def which_dataset(path):
        if "coco" in path:
            return "coco"
        elif "lvis" in path:
            return "lvis"
        else:
            None

    return which_backbone(path), which_neck(path), which_dataset(path)


for file in files_which_we_have:
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


# print(all_combis)

for file in files_which_we_have:
    backbone, neck, dataset = which(file)
    if backbone and neck and dataset:
        all_combis[(neck, backbone, dataset)] = True

print(all_combis)
if "configs" in all_combis.keys():
    print(f"In keys {True}")
if None in all_combis.values():
    print(f"In values {True}")


# def num_classes_changer(d, prefix=""):
#     for key in d:
#         full_key = f"{prefix}.{key}" if prefix else key
#         if "num_classes" in full_key:
#             print(f"Found key: {full_key}")
#             cfg._cfg_dict[full_key] = 20
#         if isinstance(d[key], dict):
#             num_classes_changer(d[key], full_key)
def num_classes_changer(d, prefix=""):
    # Use list(d.keys()) to avoid modifying the dictionary during iteration
    for key in list(d.keys()):
        full_key = f"{prefix}.{key}" if prefix else key
        if "num_classes" in full_key:
            print(f"Found key: {full_key}")
        # If the value is a dictionary, apply the function recursively
        if isinstance(d[key], dict):
            num_classes_changer(d[key], full_key)


#! Take in more files
reference_configs = {
    "double_heads": "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    "dynamic_rcnn": "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
    #! added some but not sure if right
    "fast_rcnn": "./mmdetection/configs/fast_rcnn/fast-rcnn_r50_fpn_2x_coco.py",
    "faster_rcnn": "./mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py",
    "rpn": "./mmdetection/configs/rpn/rpn_r50_fpn_2x_coco.py",
    "retinanet": "./mmdetection/configs/retinanet/retinanet_r50_fpn_ms-640-800-3x_coco.py",
    "cascade_rcnn": "./mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_20e_coco.py",
    # "cascade_rcnn": "./mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_ms-3x_coco.py",
    "grid_rcnn": "./mmdetection/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py",
    "fsaf": "./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py",
    "libra_rcnn": "./mmdetection/configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py",
    "reppoints": "./mmdetection/configs/reppoints/reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py",
    "free_anchor": "./mmdetection/configs/free_anchor/freeanchor_r50_fpn_1x_coco.py",
    "foveabox": "./mmdetection/configs/foveabox/fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
    # "double_heads": "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    "atss": "./mmdetection/configs/atss/atss_r50_fpn_1x_coco.py",
    "sabl": "./mmdetection/configs/sabl/sabl-cascade-rcnn_r50_fpn_1x_coco.py",
    # "dynamic_rcnn": "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
    "detr": "./mmdetection/configs/detr/detr_r50_8xb2-150e_coco.py",
    "paa": "./mmdetection/configs/paa/paa_r50_fpn_ms-3x_coco.py",
    "vfnet": "./mmdetection/configs/vfnet/vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco.py",
    "sparse_rcnn": "./mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
    "deformable_detr": "./mmdetection/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py",
    "tood": "./mmdetection/configs/tood/tood_r50_fpn_ms-2x_coco.py",
    "ddod": "./mmdetection/configs/ddod/ddod_r50_fpn_1x_coco.py",
    "conditional_detr": "./mmdetection/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
    "dab_detr": "./mmdetection/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py",
    "dino": "./mmdetection/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py",
    "ddq": "./mmdetection/configs/ddq/ddq-detr-5scale_r50_8xb2-12e_coco.py",
    "DiffusionDet": "./mmdetection/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py",
    "codino": "./mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py",
    # "fast_rcnn": "./mmdetection/configs/fast_rcnn/fast-rcnn_r101_fpn_2x_coco.py",
    # "faster_rcnn": "./mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_ms-3x_coco.py",
    # "rpn": "./mmdetection/configs/rpn/rpn_r101_fpn_2x_coco.py",
    # "retinanet": "./mmdetection/configs/retinanet/retinanet_r101_fpn_ms-640-800-3x_coco.py",
    # "cascade_rcnn": "./mmdetection/configs/cascade_rcnn/cascade-rcnn_r101_fpn_20e_coco.py",
    # "cascade_rcnn": "./mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_r101_fpn_ms-3x_coco.py",
    # "grid_rcnn": "./mmdetection/configs/grid_rcnn/grid-rcnn_r101_fpn_gn-head_2x_coco.py",
    # "fsaf": "./mmdetection/configs/fsaf/fsaf_r101_fpn_1x_coco.py",
    # "libra_rcnn": "./mmdetection/configs/libra_rcnn/libra-faster-rcnn_r101_fpn_1x_coco.py",
    # "reppoints": "./mmdetection/configs/reppoints/reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py",
    # "free_anchor": "./mmdetection/configs/free_anchor/freeanchor_r101_fpn_1x_coco.py",
    # "foveabox": "./mmdetection/configs/foveabox/fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
    # "atss": "./mmdetection/configs/atss/atss_r101_fpn_1x_coco.py",
    # "sabl": "./mmdetection/configs/sabl/sabl-cascade-rcnn_r101_fpn_1x_coco.py",
    # "paa": "./mmdetection/configs/paa/paa_r101_fpn_ms-3x_coco.py",
    # "vfnet": "./mmdetection/configs/vfnet/vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco.py",
    # "sparse_rcnn": "./mmdetection/configs/sparse_rcnn/sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
    # "tood": "./mmdetection/configs/tood/tood_r101_fpn_ms-2x_coco.py",#
    "centernet": "./mmdetection/configs/centernet/centernet-update_r50_fpn_8xb8-amp-lsj-200e_coco.py",
    # "fcos": "./mmdetection/configs/fcos/fcos_r50_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py",
    # "./mmdetection/configs/centernet": "./mmdetection/configs/centernet/centernet-update_r101_fpn_8xb8-amp-lsj-200e_coco.py",
    # "./mmdetection/configs/fcos": "./mmdetection/configs/fcos/fcos_r101_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py",
    "yolo": "./mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
    # "cornernet": "./mmdetection/configs/cornernet/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco.py", #! has none neck
    "guided_anchoring": "./mmdetection/configs/guided_anchoring/ga-faster-rcnn_x101-64x4d_fpn_1x_coco.py",
    # "centripetalnet": "./mmdetection/configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py", #! has none neck
    "yolox": "./mmdetection/configs/yolox/yolox_x_8xb8-300e_coco.py",
    "rtmdet": "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py",
    # "./mmdetection/configs/rtmdet": "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py",
    "glip": "./mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
    "EfficientDet": "./mmdetection/projects/EfficientDet/configs/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco.py",
    # "ViTDet": "./mmdetection/projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py",
    "Detic": "./mmdetection/projects/Detic/configs/detic_centernet2_swin-b_fpn_4x_lvis-coco-in21k.py",
}


missing_refrences = set()

for (neck, backbone, dataset), found in all_combis.items():
    if found:
        continue

    if neck in reference_configs.keys():
        # if reference_configs[neck] == "configs":
        #     neck = "EfficientDet"

        reference_file = reference_configs[neck]
        backbone_ref, neck_ref, dataset_ref = which(reference_file)
        cfg = Config.fromfile(reference_file)

        assert (
            neck == neck_ref
        ), f"Neck mismatch: {neck} != {neck_ref}, make neck in refercence list"

        if backbone != backbone_ref:
            cfg.model.backbone = new_backbone_configs[backbone]
            # print(f"{neck, backbone, dataset}")
            if neck == "libra_rcnn":
                cfg.model.neck[0].in_channels = new_neck_configs[backbone][
                    "in_channels"
                ]
            else:
                cfg.model.neck.in_channels = new_neck_configs[backbone]["in_channels"]

        # data_root as voc0712_data_root,
        # dataset_type as voc0712_dataset_type,
        # train_pipeline as voc0712_train_pipeline,
        # test_pipeline as voc0712_test_pipeline,
        # train_dataloader as voc0712_train_dataloader,
        # val_dataloader as voc0712_val_dataloader

        if dataset != dataset_ref:
            cfg.data_root = voc0712_data_root
            cfg.dataset_type = voc0712_dataset_type

            cfg.METAINFO = voc0712_METAINFO

            cfg.train_pipeline = voc0712_train_pipeline
            cfg.test_pipeline = voc0712_test_pipeline

            cfg.train_dataloader = voc0712_train_dataloader
            cfg.val_dataloader = voc0712_val_dataloader
            cfg.test_dataloader = voc0712_val_dataloader

            if cfg.train_cfg.type == "EpochBasedTrainLoop":
                cfg.train_cfg.max_epochs = cfg.train_cfg.max_epochs // 3

            num_classes_changer(cfg._cfg_dict)

            #! when try to update the metric
            # cfg.val_evaluator = (
            #     dict(
            #         ann_file="data/voc_coco_format/voc0712_val.json",
            #         backend_args=None,
            #         format_only=False,
            #         metric="bbox",
            #         type="CocoMetric",
            #     ),
            # )
            # cfg.test_evaluator = dict(
            #     ann_file="data/voc_coco_format/voc07_test.json",
            #     backend_args=None,
            #     format_only=False,
            #     metric="bbox",
            #     type="CocoMetric",
            # )

        cfg.auto_scale_lr.enable = True  #! test if works
        #! put in for real training
        # cfg.visualizer.vis_backends[0].type = "WandbVisBackend"
        # cfg.visualizer.vis_backends[0].init_kwargs = dict(
        #     project=f"{neck}_{backbone}_{dataset}"
        # )

        destination_file = os.path.join(
            "./configs_to_train", f"{neck}_{backbone}_{dataset}.py"
        )
        # destination_file = os.path.join(
        #     "./configs_to_train", f"{neck}_{cfg.model.backbone.type}_{dataset}.py"
        # )
        cfg.dump(destination_file)
        all_combis[(neck, backbone, dataset)] = True
    else:
        # print(f"Missing reference for {neck}")
        missing_refrences.add(neck)


# print(all_combis)
print(
    f"{sum(all_combis.values())} files created, {len(all_combis) - sum(all_combis.values())} missing"
)
print(f"References are needed for {missing_refrences}")


folder_path = "./configs_to_train"
files = os.listdir(folder_path)

for file in files:
    if "None" in file:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
