from torch.utils.data import Dataset

from dataloader import get_dataset

dataset_name = 'sceneflow'
data_path = "/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0"

dataset_train = get_dataset(dataset_name, data_path, "STTR", "train", )
dataset_validation = get_dataset(dataset_name, data_path, "STTR", "validation", )
dataset_test = get_dataset(dataset_name, data_path, "STTR", "test")


def check_dataset(dataset: Dataset):
    for i in range(len(dataset)):
        print('Position at: ', i)
        item = dataset.__getitem__(i)


check_dataset(dataset_train)
print("checked train")
check_dataset(dataset_validation)
print("checked validation")
check_dataset(dataset_test)