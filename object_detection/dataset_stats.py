import json
from pathlib import Path
import argparse


def count_coco_examples(ann_file):
    """
    Counts the number of images in a COCO annotation file.

    Args:
        ann_file (str or Path): Path to the COCO annotation JSON file.

    Returns:
        int: Number of images.
    """
    ann_file = Path(ann_file)
    if not ann_file.is_file():
        print(f"[Error] COCO annotation file not found: {ann_file}")
        return 0

    try:
        with ann_file.open("r") as f:
            data = json.load(f)
        num_images = len(data.get("images", []))
        return num_images
    except json.JSONDecodeError:
        print(f"[Error] JSON decoding failed for COCO annotation file: {ann_file}")
        return 0


def count_voc_examples(txt_file):
    """
    Counts the number of images listed in a VOC ImageSets txt file.

    Args:
        txt_file (str or Path): Path to the VOC ImageSets txt file.

    Returns:
        int: Number of images.
    """
    txt_file = Path(txt_file)
    if not txt_file.is_file():
        print(f"[Error] VOC ImageSets file not found: {txt_file}")
        return 0

    try:
        with txt_file.open("r") as f:
            lines = f.readlines()
        # Each line corresponds to an image
        num_images = len(lines)
        return num_images
    except Exception as e:
        print(f"[Error] Failed to read VOC ImageSets file {txt_file}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Count COCO and VOC training/testing examples."
    )

    # Define COCO paths based on your configuration
    coco_data_root = Path("data/coco/")
    coco_train_ann = coco_data_root / "annotations" / "instances_train2017.json"
    coco_val_ann = coco_data_root / "annotations" / "instances_val2017.json"

    # Define VOC paths based on your configuration
    voc_data_root = Path("data/VOCdevkit/")
    voc_years = ["VOC2007", "VOC2012"]

    # VOC2007 paths
    voc2007_root = voc_data_root / "VOC2007"
    voc2007_trainval_txt = voc2007_root / "ImageSets" / "Main" / "trainval.txt"
    voc2007_test_txt = voc2007_root / "ImageSets" / "Main" / "test.txt"

    # VOC2012 paths
    voc2012_root = voc_data_root / "VOC2012"
    voc2012_trainval_txt = voc2012_root / "ImageSets" / "Main" / "trainval.txt"

    print("=== Counting COCO Dataset Examples ===")
    # COCO Training
    num_coco_train = count_coco_examples(coco_train_ann)
    print(f"COCO Training examples: {num_coco_train}")

    # COCO Testing/Validation
    num_coco_val = count_coco_examples(coco_val_ann)
    print(f"COCO Testing/Validation examples: {num_coco_val}\n")

    print("=== Counting VOC Dataset Examples ===")
    # VOC2007 Training
    num_voc2007_trainval = count_voc_examples(voc2007_trainval_txt)
    print(f"VOC2007 Training examples: {num_voc2007_trainval}")

    # VOC2012 Training
    num_voc2012_trainval = count_voc_examples(voc2012_trainval_txt)
    print(f"VOC2012 Training examples: {num_voc2012_trainval}")

    # Total VOC Training
    total_voc_train = num_voc2007_trainval + num_voc2012_trainval
    print(f"Total VOC Training examples (VOC2007 + VOC2012): {total_voc_train}")

    # VOC2007 Testing
    num_voc2007_test = count_voc_examples(voc2007_test_txt)
    print(f"VOC2007 Testing examples: {num_voc2007_test}")


if __name__ == "__main__":
    main()
