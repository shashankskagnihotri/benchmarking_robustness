import os

error_log_path = "slurm/log_errors"
error_log_folder = os.listdir(error_log_path)


other_error_files = []

for file in error_log_folder:
    with open(f"{error_log_path}/{file}", "r") as f:
        if "CUDA out of memory" not in f.read():
            other_error_files.append(file)

cuda_error_files = []


for file in error_log_folder:
    with open(f"{error_log_path}/{file}", "r") as f:
        if "CUDA out of memory" in f.read():
            cuda_error_files.append(file)


print(f"Number of files with errors: {len(error_log_folder)}")
print(f"Number of files with other errors: {len(other_error_files)}")
print(other_error_files)
print(f"Number of files with CUDA out of memory error: {len(cuda_error_files)}")
print(cuda_error_files)


print()


# Number of files with CUDA out of memory error: 27
# ['double_heads_convnext-b_coco.err', 'codino_convnext-b_coco.err', 'rtmdet_swin-b_coco.err', 'EfficientDet_convnext-b_coco.err', 'dab_detr_convnext-b_coco.err', 'double_heads_convnext-b_voc0712.err', 'ddq_convnext-b_coco.err', 'ddq_convnext-b_voc0712.err', 'DiffusionDet_convnext-b_voc0712.err', 'DiffusionDet_convnext-b_coco.err', 'dino_swin-b_coco.err', 'dino_swin-b_voc0712.err', 'deformable_detr_convnext-b_coco.err', 'detr_swin-b_coco.err', 'glip_convnext-b_coco.err', 'conditional_detr_convnext-b_coco.err', 'yolox_r101_coco.err', 'glip_swin-b_coco.err', 'detr_convnext-b_coco.err', 'yolox_convnext-b_voc0712.err', 'rtmdet_r101_coco.err', 'dino_convnext-b_voc0712.err', 'codino_convnext-b_voc0712.err', 'rtmdet_r50_coco.err', 'deformable_detr_convnext-b_voc0712.err', 'dino_convnext-b_coco.err', 'EfficientDet_convnext-b_voc0712.err']


# 'fast_rcnn_swin-b_coco.err', 'fast_rcnn_r101_voc0712.err', 'faster_rcnn_swin-b_coco.err', 'glip_r50_voc0712.err', 'glip_swin-b_voc0712.err', 'Detic_new_r101_coco.err', 'centernet_r50_voc0712.err', 'fast_rcnn_convnext-b_voc0712.err', 'Detic_new_r101_voc0712.err', 'Detic_new_r50_voc0712.err', 'Detic_new_swin-b_coco.err', 'EfficientDet_swin-b_coco.err', 'faster_rcnn_swin-b_voc0712.err', 'centernet_r101_voc0712.err', 'fast_rcnn_r50_voc0712.err', 'EfficientDet_r101_voc0712.err', 'EfficientDet_swin-b_voc0712.err', 'EfficientDet_r50_voc0712.err', 'Detic_new_r50_coco.err', 'glip_r101_voc0712.err', 'Detic_new_convnext-b_coco.err', 'fast_rcnn_convnext-b_coco.err', 'Detic_new_swin-b_voc0712.err', 'Detic_new_convnext-b_voc0712.err', 'glip_convnext-b_voc0712.err', 'fast_rcnn_swin-b_voc0712.err']

# 142 to train in total
# 16 (EfficientDet, Detic, Fast, Glip) to 20 which are for me beyond fixing
# 4 RPN for Fast RCNN
