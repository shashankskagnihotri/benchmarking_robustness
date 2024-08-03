import os
from mmengine.config import Config
from collections import defaultdict

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"


# List all files from all directories
def list_coco_files(path):
    return [f for f in os.listdir(path) if "coco" in f]


filenames_to_train = list_coco_files(path_folder_to_train)
filenames_verified = list_coco_files(path_folder_verified)
filenames_erroneous = list_coco_files(path_folder_erroneous)
filenames_to_test = list_coco_files(path_folder_to_test)

# Define the specific setup to search for
target_setup = (
    ("ann_file", "data/VOCdevkit/voc_coco_fmt_annotations/voc07_test.json"),
    ("metric", "bbox"),
    ("proposal_nums", (100, 300, 1000)),
)

# Default values
DEFAULT_PROPOSAL_NUMS = (100, 300, 1000)
DEFAULT_METRIC = "bbox"

# Create lists to store filenames where the setup is found
val_files_with_target_setup = []
test_files_with_target_setup = []


def extract_relevant_fields(evaluator):
    relevant_fields = {
        "ann_file": evaluator.get("ann_file"),
        "metric": evaluator.get("metric", DEFAULT_METRIC),
        "proposal_nums": evaluator.get("proposal_nums", DEFAULT_PROPOSAL_NUMS),
    }
    return tuple(sorted(relevant_fields.items()))


def extract_hashable_evaluator(evaluator):
    if isinstance(evaluator, dict):
        return extract_relevant_fields(evaluator)
    elif isinstance(evaluator, list):
        return tuple(extract_relevant_fields(item) for item in evaluator)
    else:
        return evaluator


def check_evaluator_setups(filename, folder_path):
    cfg = Config.fromfile(f"{folder_path}/{filename}")

    val_evaluator = extract_hashable_evaluator(cfg.val_evaluator)
    test_evaluator = extract_hashable_evaluator(cfg.test_evaluator)

    normalized_target_setup = extract_hashable_evaluator(dict(target_setup))

    if val_evaluator == normalized_target_setup:
        val_files_with_target_setup.append(filename)
    if test_evaluator == normalized_target_setup:
        test_files_with_target_setup.append(filename)


# Check files in all directories
for filename in filenames_to_train:
    check_evaluator_setups(filename, path_folder_to_train)

for filename in filenames_verified:
    check_evaluator_setups(filename, path_folder_verified)

for filename in filenames_erroneous:
    check_evaluator_setups(filename, path_folder_erroneous)

for filename in filenames_to_test:
    check_evaluator_setups(filename, path_folder_to_test)

# Print the results
print("Files with the target val evaluator setup:")
for filename in val_files_with_target_setup:
    print(filename)

print("Files with the target test evaluator setup:")
for filename in test_files_with_target_setup:
    print(filename)


# Occurrences of each unique val evaluator setup:
# (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'bbox'), ('proposal_nums', (100, 300, 1000))): 131
# (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'bbox'), ('proposal_nums', (100, 1, 10))): 4
# (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'proposal_fast'), ('proposal_nums', (100, 300, 1000))): 4
# Occurrences of each unique test evaluator setup:

# (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'bbox'), ('proposal_nums', (100, 300, 1000))): 131
# (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'bbox'), ('proposal_nums', (100, 1, 10))): 4
# (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'proposal_fast'), ('proposal_nums', (100, 300, 1000))): 4


# Setup (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'bbox'), ('proposal_nums', (100, 1, 10))):
#   rtmdet_r50_coco.py
#   rtmdet_r101_coco.py
#   rtmdet_swin-b_coco.py
#   rtmdet_convnext-b_coco.py
# Setup (('ann_file', 'data/coco/annotations/instances_val2017.json'), ('metric', 'proposal_fast'), ('proposal_nums', (100, 300, 1000))):
#   rpn_swin-b_coco.py
#   rpn_convnext-b_coco.py
#   rpn_r50_coco.py
#   rpn_r101_coco.py
