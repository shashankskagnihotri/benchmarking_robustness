# Configmaker -> config_error_fixer -> backbone_training_parameter_changer -> verifier -> trainer -> tester

# reppoint can´t be fixed in this way since its error is environtment related

import os
from mmengine.config import Config
from pudb import set_trace


from voc0712_cocofmt_reference import (
    METAINFO as voc0712_METAINFO,
)
from mmdetection.configs._base_.datasets.voc0712 import (
    data_root as voc0712_data_root,
    dataset_type as voc0712_dataset_type,
    train_pipeline as voc0712_train_pipeline,
    test_pipeline as voc0712_test_pipeline,
)


voc_train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="RepeatDataset",  #! removed repetition, since it makes the scheduling more complicated
        times=1,
        dataset=dict(
            type="ConcatDataset",
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=["dataset_type"],
            datasets=[
                dict(
                    type=voc0712_dataset_type,
                    data_root=voc0712_data_root,
                    ann_file="VOC2007/ImageSets/Main/trainval.txt",
                    data_prefix=dict(sub_data_root="VOC2007/"),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32
                    ),
                    metainfo=voc0712_METAINFO,
                    pipeline=voc0712_train_pipeline,
                    backend_args=None,
                ),
                dict(
                    type=voc0712_dataset_type,
                    data_root=voc0712_data_root,
                    ann_file="VOC2012/ImageSets/Main/trainval.txt",
                    data_prefix=dict(sub_data_root="VOC2012/"),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32
                    ),
                    metainfo=voc0712_METAINFO,
                    pipeline=voc0712_train_pipeline,
                    backend_args=None,
                ),
            ],
        ),
    ),
)


voc_val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=voc0712_dataset_type,
        data_root=voc0712_data_root,
        ann_file="VOC2007/ImageSets/Main/test.txt",
        data_prefix=dict(sub_data_root="VOC2007/"),
        test_mode=True,
        metainfo=voc0712_METAINFO,
        pipeline=voc0712_test_pipeline,
        backend_args=None,
    ),
)

voc0712_val_evaluator = dict(
    type="VOCMetric",
    metric="mAP",
    eval_mode="11points",  #! such that we are consistent with the testset since it is only available for voc07
    iou_thrs=[0.25, 0.30, 0.40, 0.50, 0.70, 0.75],
)


def namefinder(filename):
    def neck(filename):
        return filename.split("_")[0]

    def backbone(filename):
        if "swin-b" in filename:
            return "swin-b"
        elif "convnext-b" in filename:
            return "convnext-b"
        elif "r50" in filename:
            return "r50"
        elif "r101" in filename:
            return "r101"
        else:
            return "unknown-backbone"

    def dataset(filename):
        if "coco" in filename:
            return "coco"
        elif "voc" in filename:
            return "voc0712"
        else:
            return "unknown-dataset"

    return neck(filename), backbone(filename), dataset(filename)


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


#! check how to proceed with configs that where fixed by voc implementation


def process_files(source_folder):
    filenames = os.listdir(source_folder)
    filenames.sort()
    destination_folder = "./configs_to_train"
    for filename in filenames:
        print(f"Processing file: {filename} in {source_folder}")
        filepath = os.path.join(source_folder, filename)
        neck, backbone, dataset = namefinder(filename)
        destination_file = os.path.join(
            destination_folder,
            filename,  #! use whole filename instead, sort the folders first, use if ... in filename
        )
        if "codino_swin-b_voc0712" in filename:
            print(f"Condition of {filename} met")

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
            cfg.optim_wrapper.type = "OptimWrapper"

            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "codino_swin-b_coco" in filename:
            print(f"Condition of {filename} met")

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
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "codino_r50_coco" in filename:
            print(f"Condition of {filename} met")

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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "codino_r50_voc0712" in filename:
            print(f"Condition of {filename} met")

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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "codino_convnext-b_voc0712" in filename:
            print(f"Condition of {filename} met")

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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "codino_convnext-b_coco" in filename:
            print(f"Condition of {filename} met")

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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "codino_r101_coco" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "codino_r101_voc0712" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "ddq_swin-b" in filename:
            print(f"Condition of {filename} met")
            cfg = Config.fromfile(filepath)
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "Detic_swin-b_voc0712" in filename:
            print(f"Condition of {filename} met")
            cfg = Config.fromfile(filepath)

            config_keybased_value_changer(
                config_dictionary=cfg._cfg_dict,
                searched_key="num_classes",
                do_new=True,
                new_absolute_value=80,
                change_old_value_by=1,
                prefix="",
            )

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

            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                os.remove(filepath)
                cfg.dump(destination_file)

        if "EfficientDet" in filename:
            print(f"Condition of {filename} met")

            cfg = Config.fromfile(filepath)
            cfg.vis_backends = [
                dict(type="LocalVisBackend"),
            ]
            cfg.visualizer.vis_backends = [
                dict(type="LocalVisBackend"),
            ]
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "free_anchor_convnext-b" in filename:
            print(f"Condition of {filename} met")
            cfg = Config.fromfile(filepath)
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "free_anchor_swin-b" in filename:
            print(f"Condition of {filename} met")
            cfg = Config.fromfile(filepath)
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "glip_convnext-b" in filename or "glip_swin-b" in filename:
            print(f"Condition of {neck}, {backbone}, {dataset} met")
            cfg = Config.fromfile(filepath)

            if hasattr(cfg, "load_from"):
                cfg.pop("load_from", "Not found")

            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                os.remove(filepath)
                cfg.dump(destination_file)
        # elif "paa_convnext-b_voc0712" in filename:
        #     print(f"Condition of {filename} met")
        #     cfg = Config.fromfile(filepath)
        #     cfg.optim_wrapper.type = "OptimWrapper"
        #     if source_folder != destination_folder:
        #         cfg.dump(destination_file)
        #         os.remove(filepath)
        #     else:
        #         cfg.dump(destination_file)
        elif "sparse_rcnn_convnext-b" in filename or "sparse_rcnn_swin-b" in filename:
            print(f"Condition of {neck}, {backbone}, {dataset} met")
            cfg = Config.fromfile(filepath)
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "tood_convnext-b" in filename or "tood_swin-b" in filename:
            print(f"Condition of {neck}, {backbone}, {dataset} met")
            cfg = Config.fromfile(filepath)
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "vfnet_swin-b" in filename or "vfnet_convnext-b" in filename:
            print(f"Condition of {filename} met")
            cfg = Config.fromfile(filepath)
            cfg.optim_wrapper.type = "OptimWrapper"
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)
        elif "yolo_convnext-b_coco" in filename:
            print(f"Condition of {filename} met")

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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_convnext-b_voc0712" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_r50_coco" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_r50_voc0712" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_r101_coco" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_r101_voc0712" in filename:
            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_swin-b_coco" in filename:
            # set_trace()

            print(f"Condition of {filename} met")
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)

        elif "yolo_swin-b_voc0712" in filename:
            print(f"Condition of {filename} met")

            # set_trace()
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
            if source_folder != destination_folder:
                cfg.dump(destination_file)
                os.remove(filepath)
            else:
                cfg.dump(destination_file)


path_folder_to_train = "./configs_to_train"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"


process_files(path_folder_to_train)
process_files(path_folder_erroneous)
process_files(path_folder_to_test)
