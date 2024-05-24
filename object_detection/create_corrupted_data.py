import os
from tqdm import tqdm
from pycocotools.coco import COCO
from imagecorruptions import get_corruption_names, corrupt
import mmcv


def common_corruptions(dataset_type, input_dir, output_dir):
    """
    Apply common corruptions to images in COCO or Pascal VOC dataset.

    :param dataset_type: Type of dataset ('coco' or 'voc').
    :param input_dir: Path to the input dataset directory.
    :param output_dir: Path to the output directory to save corrupted images.
    """
    if dataset_type == "coco":
        # Load the COCO annotations
        ann_file = os.path.join(input_dir, "annotations", "instances_val2017.json")
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        img_dir = os.path.join(input_dir, "val2017")

    elif dataset_type == "voc":
        # Load the VOC image set file
        img_set_file = os.path.join(input_dir, "ImageSets", "Main", "val.txt")
        with open(img_set_file, "r") as f:
            img_ids = [line.strip() for line in f.readlines()]
        img_dir = os.path.join(input_dir, "JPEGImages")

    else:
        raise ValueError("dataset_type must be either 'coco' or 'voc'")

    # Apply corruptions and save to folders
    corruptions = get_corruption_names()
    severities = [1, 2, 3, 4, 5]
    for corruption in corruptions:
        for severity in severities:
            # Create folders for each corruption and severity
            output_folder = os.path.join(
                output_dir, "cc", corruption, f"severity_{severity}"
            )
            os.makedirs(output_folder, exist_ok=True)

            # Check if folder already contains all images
            existing_images = len(os.listdir(output_folder))
            if existing_images >= len(img_ids):
                print(
                    f"Skipping corruption '{corruption}' with severity {severity} as all {existing_images} images already exist."
                )
                continue

            # Apply corruption and save corrupted images
            with tqdm(
                total=len(img_ids),
                desc=f"Corruption: {corruption}, Severity: {severity}",
            ) as pbar:
                for img_id in img_ids:
                    if dataset_type == "coco":
                        img_info = coco.loadImgs(img_id)[0]
                        img_path = os.path.join(img_dir, img_info["file_name"])
                        output_path = os.path.join(output_folder, img_info["file_name"])
                    else:  # 'voc'
                        img_path = os.path.join(img_dir, f"{img_id}.jpg")
                        output_path = os.path.join(output_folder, f"{img_id}.jpg")

                    image = mmcv.imread(img_path)
                    corrupted_image = corrupt(
                        image, corruption_name=corruption, severity=severity
                    )
                    mmcv.imwrite(corrupted_image, output_path)
                    pbar.update(1)


if __name__ == "__main__":
    dataset_type = "voc"
    input_dir = "data/VOCdevkit/VOC2012/"
    output_dir = "data/VOCdevkit/VOC2012/cc/"
    common_corruptions(dataset_type, input_dir, output_dir)

    dataset_type = "voc"
    input_dir = "data/VOCdevkit/VOC2017/"
    output_dir = "data/VOCdevkit/VOC2017/cc/"
    common_corruptions(dataset_type, input_dir, output_dir)

    dataset_type = "coco"
    input_dir = "data/coco/"
    output_dir = "data/coco/cc/"
    common_corruptions(dataset_type, input_dir, output_dir)
