# import json

# val07_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/val.txt"
# test07_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"

# val07_json_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/val.json"
# test07_json_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/test.json"


# def txt_to_json(txt_path, json_path):
#     with open(txt_path, "r") as f:
#         lines = f.readlines()
#     lines = [line.strip() for line in lines]
#     with open(json_path, "w") as f:
#         for i, line in enumerate(lines):
#             data = {"image_id": line}  # Renamed key to 'image_id'
#             json.dump(data, f, indent=4)  # Write JSON with indentation for readability
#             if i < len(lines) - 1:  # Add comma if it's not the last line
#                 f.write(",")
#             f.write("\n")  # Add a new line after each JSON object


# txt_to_json(val07_path, val07_json_path)
# txt_to_json(test07_path, test07_json_path)

import json


def create_json_dataset(txt_path, json_path):
    # Read VOC2007 dataset file
    with open(txt_path, "r") as file:
        image_ids = [int(line.strip()) for line in file]

    # Create JSON structure
    data = {
        "info": {
            "description": "VOC2007 Dataset",
            "url": "your_dataset_url",
            "version": "1.0",
            "year": 2007,
            "contributor": "Your Name",
            "date_created": "current_date",
        },
        "annotations": [],
    }

    # Populate images field with image IDs
    for image_id in image_ids:
        data["annotations"].append({"image_id": image_id})

    # Write JSON data to a file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


# Paths
val07_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/val.txt"
test07_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"

val07_json_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/val.json"
test07_json_path = "./data/VOCdevkit/VOC2007/ImageSets/Main/test.json"

# Create JSON dataset for validation set
create_json_dataset(val07_path, val07_json_path)

# Create JSON dataset for test set
create_json_dataset(test07_path, test07_json_path)
