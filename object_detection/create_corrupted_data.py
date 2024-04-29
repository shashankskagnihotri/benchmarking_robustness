import os
from tqdm import tqdm
from pycocotools.coco import COCO
from imagecorruptions import get_corruption_names, corrupt
import mmcv


def common_corruptions_coco(data_dir="data/coco/"):
    # Load the COCO annotations
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)

    # Apply corruptions and save to folders
    corruptions = get_corruption_names()
    severities = [1, 2, 3, 4, 5]
    for corruption in corruptions:
        for severity in severities:
            # Create folders for each corruption and severity
            output_folder = os.path.join(
                data_dir, "cc", corruption, f"severity_{severity}"
            )
            os.makedirs(output_folder, exist_ok=True)

            # Check if folder already contains all images
            existing_images = len(os.listdir(output_folder))
            if existing_images >= len(coco.getImgIds()):
                print(
                    f"Skipping corruption '{corruption}' with severity {severity} as all {existing_images} images already exist."
                )
                continue

            # Apply corruption and save corrupted images
            with tqdm(
                total=len(coco.getImgIds()),
                desc=f"Corruption: {corruption}, Severity: {severity}",
            ) as pbar:
                for img_id in coco.getImgIds():
                    img_info = coco.loadImgs(img_id)[0]
                    img_path = os.path.join(data_dir, "val2017", img_info["file_name"])
                    image = mmcv.imread(img_path)
                    corrupted_image = corrupt(
                        image, corruption_name=corruption, severity=severity
                    )
                    output_path = os.path.join(output_folder, img_info["file_name"])
                    mmcv.imwrite(corrupted_image, output_path)
                    pbar.update(1)

def common_corruptions_3D_codo():
    

if __name__ == "__main__":
    common_corruptions_coco()
