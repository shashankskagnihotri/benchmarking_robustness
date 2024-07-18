import os
from mmengine.config import Config
from voc0712_cocofmt_reference import (
    METAINFO as voc0712_METAINFO,
    data_root as voc0712_data_root,
    dataset_type as voc0712_dataset_type,
    train_pipeline as voc0712_train_pipeline,
    test_pipeline as voc0712_test_pipeline,
    val_evaluator as voc0712_val_evaluator,
)
from rich.traceback import install

#! import the reference configuration of voc with voc metric but keep as much as possible e.g. metainfos

install()


files_which_we_have = [
    "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
    "./mmdetection/configs/ddod/ddod_r50_fpn_1x_coco.py",
    "./mmdetection/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
    "./mmdetection/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py",
    "./mmdetection/configs/fast_rcnn/fast-rcnn_r50_fpn_2x_coco.py",  #! not pretrained might have
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
    "./mmdetection/configs/reppoints/reppoints-moment_r101_fpn-gn_head-gn_2x_coco.py",
    "./mmdetection/configs/free_anchor/freeanchor_r101_fpn_1x_coco.py",
    "./mmdetection/configs/foveabox/fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py",
    "./mmdetection/configs/atss/atss_r101_fpn_1x_coco.py",
    "./mmdetection/configs/sabl/sabl-cascade-rcnn_r101_fpn_1x_coco.py",
    "./mmdetection/configs/paa/paa_r101_fpn_ms-3x_coco.py",
    "./mmdetection/configs/vfnet/vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco.py",
    "./mmdetection/configs/sparse_rcnn/sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py",
    "./mmdetection/configs/tood/tood_r101_fpn_ms-2x_coco.py",
    "./mmdetection/configs/centernet/centernet-update_r50_fpn_8xb8-amp-lsj-200e_coco.py",
    "./mmdetection/configs/centernet/centernet-update_r101_fpn_8xb8-amp-lsj-200e_coco.py",
    "./mmdetection/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py",
    "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py",
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_4xb32-100e_coco.py",
    "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py",
    "./mmdetection/projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis.py",
    "./mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
    # "./mmdetection/configs/cornernet/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco.py",
    "./mmdetection/configs/guided_anchoring/ga-faster-rcnn_x101-64x4d_fpn_1x_coco.py",
    # "./mmdetection/configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py",
    "./mmdetection/configs/yolox/yolox_x_8xb8-300e_coco.py",
    "./mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
    "./mmdetection/projects/EfficientDet/configs/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco.py",
    # "./mmdetection/projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py",
]


# for specifying the parameters
new_backbone_configs = {
    "swin-b": {  # "./mmdetection/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py"
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
    "convnext-b": {  # "./mmdetection/configs/rtmdet/rtmdet_l_convnext_b_4xb32-100e_coco.py"
        # "_delete_": True,
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
swin_b_max_epochs = 100
swin_b_base_lr = 0.001
swin_b_optim_wrapper = dict(
    optimizer=dict(lr=0.001, type="AdamW", weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type="OptimWrapper",
)
swin_b_param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=1e-05, type="LinearLR"),
    dict(
        T_max=50,
        begin=50,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=5e-05,
        type="CosineAnnealingLR",
    ),
]


convnext_max_epochs = 100


new_neck_configs = {
    "swin-b": {"in_channels": [256, 512, 1024, 2048]},
    "convnext-b": {"in_channels": [256, 512, 1024]},
    "r101": {"in_channels": [256, 512, 1024, 2048]},
    "r50": {"in_channels": [256, 512, 1024, 2048]},
}


voc_train_dataloader = dict(
    dataset=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            # _delete_=True,
            type=voc0712_dataset_type,
            data_root=voc0712_data_root,
            ann_file="voc_coco_fmt_annotations/voc0712_trainval.json",  # changed from annotations/....
            data_prefix=dict(img=""),
            metainfo=voc0712_METAINFO,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=voc0712_train_pipeline,
            backend_args=None,
        ),
    )
)
voc_val_dataloader = dict(
    dataset=dict(
        type=voc0712_dataset_type,
        data_root=voc0712_data_root,  #! was not originally  present
        ann_file="voc_coco_fmt_annotations/voc07_test.json",  # changed from annotations/....
        data_prefix=dict(img=""),
        metainfo=voc0712_METAINFO,
        pipeline=voc0712_test_pipeline,
    )
)


necks_we_want = [
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
    "Detic_new",
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
]
datasets_we_want = [
    "coco",
    "voc0712",
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
        else:
            return "voc0712"

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


all_combis = {}
for n in necks_we_want:
    for b in backbones_we_want:
        for d in datasets_we_want:
            all_combis[(n, b, d)] = False


for file in files_which_we_have:
    backbone, neck, dataset = which(file)
    if backbone and neck and dataset:
        all_combis[(neck, backbone, dataset)] = True

print(all_combis)
if "configs" in all_combis.keys():
    print(f"In keys {True}")
if None in all_combis.values():
    print(f"In values {True}")


def config_keybased_value_changer(
    config_dictionary,
    searched_key,
    do_new,
    new_absolute_value,
    change_old_value_by,
    prefix="",
):
    # If the input is a dictionary, iterate through its keys
    if isinstance(config_dictionary, dict):
        for key in list(config_dictionary.keys()):
            full_key = f"{prefix}.{key}" if prefix else key
            if key == searched_key:
                print(f"Found key: {full_key}")
                print(f"Old Value: {config_dictionary[key]}")
                if do_new:
                    config_dictionary[key] = (
                        new_absolute_value  # Set the new value for the searched key
                    )
                else:
                    config_dictionary[key] = (
                        config_dictionary[key] // change_old_value_by
                    )  # Change the old value
                print(f"New Value: {config_dictionary[key]}")
            # If the value is a dictionary, apply the function recursively
            if isinstance(config_dictionary[key], dict):
                config_keybased_value_changer(
                    config_dictionary[key],
                    searched_key,
                    do_new,
                    new_absolute_value,
                    change_old_value_by,
                    full_key,
                )
            # If the value is a list, apply the function to each element
            elif isinstance(config_dictionary[key], list):
                for i, item in enumerate(config_dictionary[key]):
                    config_keybased_value_changer(
                        item,
                        searched_key,
                        do_new,
                        new_absolute_value,
                        change_old_value_by,
                        f"{full_key}[{i}]",
                    )

    # If the input is a list, iterate through its items
    elif isinstance(config_dictionary, list):
        for i, item in enumerate(config_dictionary):
            config_keybased_value_changer(
                item,
                searched_key,
                do_new,
                new_absolute_value,
                change_old_value_by,
                f"{prefix}[{i}]",
            )


def adjust_param_scheduler(cfg, factor=3):
    if (
        hasattr(cfg, "model")
        and hasattr(cfg.model, "optim_wrapper")
        and hasattr(cfg.model.optim_wrapper, "param_scheduler")
    ):
        print("Found param_scheduler in configuration.")

        for param_scheduler in cfg.model.optim_wrapper.param_scheduler:
            if hasattr(param_scheduler, "by_epoch"):
                if hasattr(param_scheduler, "begin"):
                    print(f"Old begin: {param_scheduler.begin}")
                    param_scheduler.begin = param_scheduler.begin // factor
                    print(f"New begin: {param_scheduler.begin}")

                if hasattr(param_scheduler, "end"):
                    print(f"Old end: {param_scheduler.end}")
                    param_scheduler.end = param_scheduler.end // factor
                    print(f"New end: {param_scheduler.end}")

                if hasattr(param_scheduler, "milestones"):
                    print(f"Old milestones: {param_scheduler.milestones}")
                    param_scheduler.milestones = [
                        milestone // factor for milestone in param_scheduler.milestones
                    ]
                    print(f"New milestones: {param_scheduler.milestones}")
    elif hasattr(cfg, "param_scheduler"):
        print("Found param_scheduler in configuration.")

        for param_scheduler in cfg.param_scheduler:
            if hasattr(param_scheduler, "by_epoch"):
                if hasattr(param_scheduler, "begin"):
                    print(f"Old begin: {param_scheduler.begin}")
                    param_scheduler.begin = param_scheduler.begin // factor
                    print(f"New begin: {param_scheduler.begin}")

                if hasattr(param_scheduler, "end"):
                    print(f"Old end: {param_scheduler.end}")
                    param_scheduler.end = param_scheduler.end // factor
                    print(f"New end: {param_scheduler.end}")

                if hasattr(param_scheduler, "milestones"):
                    print(f"Old milestones: {param_scheduler.milestones}")
                    param_scheduler.milestones = [
                        milestone // factor for milestone in param_scheduler.milestones
                    ]
                    print(f"New milestones: {param_scheduler.milestones}")
    else:
        print("param_scheduler not found in configuration.")


def dataset_assigner(
    cfg,
    data_root,
    dataset_type,
    train_pipeline,
    test_pipeline,
    train_dataloader,
    val_dataloader,
    val_evaluator,
):
    cfg.data_root = data_root
    cfg.dataset_type = dataset_type

    cfg.train_pipeline = train_pipeline
    cfg.test_pipeline = test_pipeline

    cfg.train_dataloader = train_dataloader
    cfg.val_dataloader = val_dataloader
    cfg.test_dataloader = val_dataloader

    cfg.val_evaluator = val_evaluator
    cfg.test_evaluator = val_evaluator


reference_configs = {
    "double_heads": "./mmdetection/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py",
    "dynamic_rcnn": "./mmdetection/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py",
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
    "Detic_new": "./mmdetection/projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis_in21k-lvis.py",
}


missing_refrences = set()

for (neck, backbone, dataset), found in all_combis.items():
    if found:
        continue

    if neck in reference_configs.keys():
        reference_file = reference_configs[neck]
        backbone_ref, neck_ref, dataset_ref = which(reference_file)
        cfg = Config.fromfile(reference_file)

        assert (
            neck == neck_ref
        ), f"Neck mismatch: {neck} != {neck_ref}, make neck in refercence list"

        if neck == "Detic_new" and dataset == "coco":
            config_keybased_value_changer(
                config_dictionary=cfg._cfg_dict,
                searched_key="num_classes",
                do_new=True,
                new_absolute_value=80,
                change_old_value_by=1,
                prefix="",
            )
            #! put in a reference for the coco dataset
            # dataset_assigner(
            #     cfg,
            #     coco_data_root,
            #     coco_dataset_type,
            #     coco_train_pipeline,
            #     coco_test_pipeline,
            #     coco_train_dataloader,
            #     coco_val_dataloader,
            #     coco_val_evaluator,
            # )

        elif neck == "Detic_new" and dataset == "voc0712":
            config_keybased_value_changer(
                config_dictionary=cfg._cfg_dict,
                searched_key="num_classes",
                do_new=True,
                new_absolute_value=20,
                change_old_value_by=1,
                prefix="",
            )
            original_train_batch_size = cfg.train_dataloader.batch_size
            original_val_batch_size = cfg.val_dataloader.batch_size
            original_test_batch_size = cfg.test_dataloader.batch_size

            dataset_assigner(
                cfg,
                voc0712_data_root,
                voc0712_dataset_type,
                voc0712_train_pipeline,
                voc0712_test_pipeline,
                voc_train_dataloader,
                voc_val_dataloader,
                voc0712_val_evaluator,
            )

            cfg.train_dataloader.batch_size = original_train_batch_size
            cfg.val_dataloader.batch_size = original_val_batch_size
            cfg.test_dataloader.batch_size = original_test_batch_size

            config_keybased_value_changer(
                config_dictionary=cfg._cfg_dict,
                searched_key="num_classes",
                do_new=True,
                new_absolute_value=20,
                change_old_value_by=1,
                prefix="",
            )

            if cfg.train_cfg.type == "EpochBasedTrainLoop":
                config_keybased_value_changer(
                    config_dictionary=cfg._cfg_dict,
                    searched_key="max_epochs",
                    do_new=False,
                    new_absolute_value=0,
                    change_old_value_by=3,
                    prefix="",
                )
                adjust_param_scheduler(cfg, factor=3)

                config_keybased_value_changer(
                    config_dictionary=cfg._cfg_dict,
                    searched_key="val_interval",
                    do_new=False,
                    new_absolute_value=0,
                    change_old_value_by=3,
                    prefix="",
                )

        elif neck == "Yolo":
            pass

        else:
            if backbone != backbone_ref:
                cfg.model.backbone = new_backbone_configs[backbone]
                if neck == "libra_rcnn":
                    cfg.model.neck[0].in_channels = new_neck_configs[backbone][
                        "in_channels"
                    ]
                else:
                    cfg.model.neck.in_channels = new_neck_configs[backbone][
                        "in_channels"
                    ]

            if dataset != dataset_ref:
                #! the voc reference didn´t have a batchsize now we use the batchsize of the model reference
                original_train_batch_size = cfg.train_dataloader.batch_size
                original_val_batch_size = cfg.val_dataloader.batch_size
                original_test_batch_size = cfg.test_dataloader.batch_size

                dataset_assigner(
                    cfg,
                    voc0712_data_root,
                    voc0712_dataset_type,
                    voc0712_train_pipeline,
                    voc0712_test_pipeline,
                    voc_train_dataloader,
                    voc_val_dataloader,
                    voc0712_val_evaluator,
                )

                cfg.train_dataloader.batch_size = original_train_batch_size
                cfg.val_dataloader.batch_size = original_val_batch_size
                cfg.test_dataloader.batch_size = original_test_batch_size

                config_keybased_value_changer(
                    config_dictionary=cfg._cfg_dict,
                    searched_key="num_classes",
                    do_new=True,
                    new_absolute_value=20,
                    change_old_value_by=1,
                    prefix="",
                )

                if cfg.train_cfg.type == "EpochBasedTrainLoop":
                    config_keybased_value_changer(
                        config_dictionary=cfg._cfg_dict,
                        searched_key="max_epochs",
                        do_new=False,
                        new_absolute_value=0,
                        change_old_value_by=3,
                        prefix="",
                    )
                    adjust_param_scheduler(cfg, factor=3)

                    config_keybased_value_changer(
                        config_dictionary=cfg._cfg_dict,
                        searched_key="val_interval",
                        do_new=False,
                        new_absolute_value=0,
                        change_old_value_by=3,
                        prefix="",
                    )

        #! put in for real in training and testing!!!
        # cfg.visualizer.vis_backends[0].type = "WandbVisBackend"
        # cfg.visualizer.vis_backends[0].init_kwargs = dict(
        #     project=f"{neck}_{backbone}_{dataset}"
        # )

        destination_file = os.path.join(
            "./configs_to_train", f"{neck}_{backbone}_{dataset}.py"
        )
        cfg.dump(destination_file)
        all_combis[(neck, backbone, dataset)] = True

    else:
        missing_refrences.add(neck)


print(
    f"{sum(all_combis.values())} files created, {len(all_combis) - sum(all_combis.values())} missing"
)
print(f"References are needed for {missing_refrences}")


training_folder_path = "./configs_to_train"
files = os.listdir(training_folder_path)

for file in files:
    if "None" in file:
        file_path = os.path.join(training_folder_path, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        cfg = Config.fromfile(os.path.join(training_folder_path, file))
        if hasattr(cfg, "auto_scale_lr"):
            cfg.auto_scale_lr.enable = True
        else:
            cfg.auto_scale_lr = dict(enable=True)
            cfg.auto_scale_lr.base_batch_size = 16


test_folder_path = "./configs_to_test"
test_files = os.listdir(test_folder_path)

for test_file in test_files:
    cfg = Config.fromfile(os.path.join(test_folder_path, test_file))
    if hasattr(cfg, "auto_scale_lr"):
        cfg.auto_scale_lr.enable = True
    else:
        cfg.auto_scale_lr = dict(enable=True)
        cfg.auto_scale_lr.base_batch_size = 16


#! Please remember to check the bottom of the specific config file you want to use, it will have auto_scale_lr.base_batch_size if the batch size is not 16. If you can’t find those values, check the config file which in _base_=[xxx] and you will find it. Please do not modify its values if you want to automatically scale the LR.
