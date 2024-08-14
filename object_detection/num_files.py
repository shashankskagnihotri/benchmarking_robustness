import os


def get_filenames(path_folder):
    """Get list of filenames in the specified folder."""
    return os.listdir(path_folder)


def count_backbone_files(path_folder, backbone):
    """Count and print details of files for the specified backbone in the given folder."""
    filenames = get_filenames(path_folder)

    num_backbone_files = 0
    num_voc_backbone_files = 0
    num_coco_backbone_files = 0

    backbone_file_names = []
    voc_backbone_file_names = []
    coco_backbone_file_names = []

    for filename in filenames:
        if backbone in filename:
            num_backbone_files += 1
            backbone_file_names.append(filename)
            if "voc" in filename:
                num_voc_backbone_files += 1
                voc_backbone_file_names.append(filename)
            elif "coco" in filename:
                num_coco_backbone_files += 1
                coco_backbone_file_names.append(filename)

    print(f"Num of voc {backbone} files in {path_folder}:", num_voc_backbone_files)
    print(f"Num of coco {backbone} files in {path_folder}:", num_coco_backbone_files)
    print(f"Num of {backbone} files in {path_folder}:", num_backbone_files)

    backbone_file_names.sort()
    voc_backbone_file_names.sort()
    coco_backbone_file_names.sort()

    print(f"{backbone.capitalize()} files:")
    for filename in backbone_file_names:
        print(filename)

    print(f"VOC {backbone.capitalize()} files:")
    for filename in voc_backbone_file_names:
        print(filename)

    print(f"COCO {backbone.capitalize()} files:")
    for filename in coco_backbone_file_names:
        print(filename)


path_folder_to_train = "./configs_to_train"
# path_folder_verified = "./configs_verified"
# path_folder_erroneous = "./configs_erroneous/verification"
# path_folder_to_test = "./configs_to_test"


backbones = ["convnext", "swin"]


for folder in [
    path_folder_to_train,
    # path_folder_verified,
    # path_folder_erroneous,
    # path_folder_to_test,
]:
    for backbone in backbones:
        print(f"Processing {backbone} files in {folder}")
        count_backbone_files(folder, backbone)
        print("\n")
