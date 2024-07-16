# Configmaker -> config_error_fixer -> backbone_training_parameter_changer -> verifier -> trainer -> tester

# reppoint can´t be fixed in this way since its error is environtment related

import os
from mmengine.config import Config
from config_maker import which

#! check how to proceed with configs that where fixed by voc implementation

path_folder_to_train = "./configs_to_train"
path_folder_erroneous = "./configs_erroneous/verification"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_erroneous = os.listdir(path_folder_erroneous)


for filename in filenames_to_train:
    filepath = os.join(path_folder_to_train, filename)
    neck, backbone, dataset = which(filepath)
    destination_file = os.path.join(
        "./configs_to_train", f"{neck}_{backbone}_{dataset}.py"
    )

    if neck == "atss" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "atss" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "cascade_rcnn" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "centernet" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "centernet" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "centernet" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "codino" and backbone == "swin-b" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 12
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=12,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.model.data_preprocessor = dict(
            batch_augments=[
                # dict(
                #     pad_mask=True,
                #     size=(
                #         1024,
                #         1024,
                #     ),
                #     type="BatchFixedSizePad",
                # ),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_mask=True,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type="DetDataPreprocessor",
        )
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "swin-b" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 36
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=36,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "r50" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 36
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=36,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "r50" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 12
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=12,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.model.data_preprocessor = dict(
            batch_augments=[
                # dict(
                #     pad_mask=True,
                #     size=(
                #         1024,
                #         1024,
                #     ),
                #     type="BatchFixedSizePad",
                # ),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_mask=True,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type="DetDataPreprocessor",
        )
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "convnext-b" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 12
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=12,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.model.data_preprocessor = dict(
            batch_augments=[
                # dict(
                #     pad_mask=True,
                #     size=(
                #         1024,
                #         1024,
                #     ),
                #     type="BatchFixedSizePad",
                # ),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_mask=True,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type="DetDataPreprocessor",
        )
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "convnext-b" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 36
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=36,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "r101" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 36
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=36,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.dump(destination_file)

    elif neck == "codino" and backbone == "r101" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.max_epochs = 12
        cfg.param_scheduler = [
            dict(
                type="MultiStepLR",
                begin=0,
                end=12,
                by_epoch=True,
                milestones=[10],
                gamma=0.1,
            )
        ]
        cfg.model.data_preprocessor = dict(
            batch_augments=[
                # dict(
                #     pad_mask=True,
                #     size=(
                #         1024,
                #         1024,
                #     ),
                #     type="BatchFixedSizePad",
                # ),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_mask=True,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type="DetDataPreprocessor",
        )
        cfg.dump(destination_file)

    elif neck == "ddod" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "Detic_new" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "Detic_new" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "Detic_new" and backbone == "r101" and dataset == "coco":
        pass
    elif neck == "Detic_new" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "Detic_new" and backbone == "r50" and dataset == "coco":
        pass
    elif neck == "Detic_new" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "Detic_new" and backbone == "swin-b" and dataset == "coco":
        pass
    elif neck == "Detic_new" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "double_heads" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "dynamic_rcnn" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "dynamic_rcnn" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "dynamic_rcnn" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "dynamic_rcnn" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "dynamic_rcnn" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "EfficientDet" and backbone == "r101" and dataset == "coco":
        pass
    elif neck == "EfficientDet" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "EfficientDet" and backbone == "r50" and dataset == "coco":
        pass
    elif neck == "EfficientDet" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "EfficientDet" and backbone == "swin-b" and dataset == "coco":
        pass
    elif neck == "EfficientDet" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "EfficientDet" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "fast_rcnn" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "fast_rcnn" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "fast_rcnn" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "fast_rcnn" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "fast_rcnn" and backbone == "swin-b" and dataset == "coco":
        pass
    elif neck == "fast_rcnn" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "faster_rcnn" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "foveabox" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "glip" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "glip" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "glip" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "glip" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "grid_rcnn" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "grid_rcnn" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "guided_anchoring" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "libra_rcnn" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "paa" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "reppoints" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "reppoints" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "reppoints" and backbone == "r101" and dataset == "coco":
        pass
    elif neck == "reppoints" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "reppoints" and backbone == "r50" and dataset == "coco":
        pass
    elif neck == "reppoints" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "reppoints" and backbone == "swin-b" and dataset == "coco":
        pass
    elif neck == "reppoints" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "rtmdet" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "rtmdet" and backbone == "r50" and dataset == "coco":
        pass
    elif neck == "rtmdet" and backbone == "r50" and dataset == "voc0712":
        pass
    elif neck == "rtmdet" and backbone == "r101" and dataset == "coco":
        pass
    elif neck == "rtmdet" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "rtmdet" and backbone == "swin-b" and dataset == "coco":
        pass
    elif neck == "rtmdet" and backbone == "swin-b" and dataset == "voc0712":
        pass
    elif neck == "tood" and backbone == "convnext-b" and dataset == "coco":
        pass
    elif neck == "tood" and backbone == "convnext-b" and dataset == "voc0712":
        pass
    elif neck == "tood" and backbone == "r101" and dataset == "voc0712":
        pass
    elif neck == "yolo" and backbone == "convnext-b" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            3,
            2,
            1,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "convnext-b" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            3,
            2,
            1,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "r50" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            2,
            1,
            0,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "r50" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            2,
            1,
            0,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "r101" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            2,
            1,
            0,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "r101" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            2,
            1,
            0,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "swin-b" and dataset == "coco":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            3,
            2,
            1,
        ]
        cfg.model.backbone.depths = [
            2,
            2,
            18,
            2,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolo" and backbone == "swin-b" and dataset == "voc0712":
        cfg = Config.fromfile(filepath)
        cfg.model.backbone.out_indices = [
            3,
            2,
            1,
        ]
        cfg.model.backbone.depths = [
            2,
            2,
            18,
            2,
        ]
        cfg.model.neck.in_channels = [
            1024,
            512,
            256,
        ]
        cfg.dump(destination_file)

    elif neck == "yolox" and backbone == "r101" and dataset == "coco":
        pass
    elif neck == "yolox" and backbone == "swin-b" and dataset == "coco":
        pass
    elif neck == "yolox" and backbone == "swin-b" and dataset == "voc0712":
        pass
