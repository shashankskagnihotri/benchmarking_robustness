import os
from mmengine.config import Config

path_folder_to_train = "./configs_to_train"
path_folder_verified = "./configs_verified"
path_folder_erroneous = "./configs_erroneous/verification"
path_folder_to_test = "./configs_to_test"

filenames_to_train = os.listdir(path_folder_to_train)
filenames_verified = os.listdir(path_folder_verified)
filenames_erroneous = os.listdir(path_folder_erroneous)
filenames_to_test = os.listdir(path_folder_to_test)

# Global variables to track if all VOC configurations are equal
voc_metric_equal = True
voc_eval_mode_equal = True
voc_iou_thrs_equal = True
voc_num_proposals_equal = True

# Lists to track non-equal configurations
list_non_equal_voc_metric = []
list_non_equal_voc_eval_mode = []
list_non_equal_voc_iou_thrs = []
list_non_equal_voc_num_proposals = []


def check_voc_metric(filename, folder_path):
    global \
        voc_metric_equal, \
        voc_eval_mode_equal, \
        voc_iou_thrs_equal, \
        voc_num_proposals_equal
    if "voc" in filename.lower():
        cfg = Config.fromfile(os.path.join(folder_path, filename))
        print(f"Checking file: {filename} in {folder_path}")

        if cfg.val_evaluator.metric != "mAP":
            voc_metric_equal = False
            list_non_equal_voc_metric.append(filename)

        if cfg.val_evaluator.eval_mode != "11points":
            voc_eval_mode_equal = False
            list_non_equal_voc_eval_mode.append(filename)

        if cfg.val_evaluator.iou_thrs != [0.25, 0.30, 0.40, 0.50, 0.70, 0.75]:
            voc_iou_thrs_equal = False
            list_non_equal_voc_iou_thrs.append(filename)

        # check if has attribute num_proposals
        if hasattr(cfg.val_evaluator, "num_proposals"):
            print(
                f"{filename} from {folder_path} has num_proposals {cfg.val_evaluator.num_proposals}"
            )
            if cfg.val_evaluator.num_proposals != (100, 300, 1000):
                voc_num_proposals_equal = False
                list_non_equal_voc_num_proposals.append(filename)
            voc_num_proposals_equal = False
            list_non_equal_voc_num_proposals.append(filename)


# Check files in all directories
for filename in filenames_verified:
    check_voc_metric(filename, path_folder_verified)
for filename in filenames_erroneous:
    check_voc_metric(filename, path_folder_erroneous)
for filename in filenames_to_test:
    check_voc_metric(filename, path_folder_to_test)
for filename in filenames_to_train:
    check_voc_metric(filename, path_folder_to_train)

print("Differing VOC metric:", list_non_equal_voc_metric)
print("Differing VOC eval mode:", list_non_equal_voc_eval_mode)
print("Differing VOC iou thrs:", list_non_equal_voc_iou_thrs)
print("Differing VOC num proposals:", list_non_equal_voc_num_proposals)
