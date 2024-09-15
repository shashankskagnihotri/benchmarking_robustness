from platform import architecture

from dataloader import get_dataset
from torch.utils.data import DataLoader
import pudb

# Dataset

datapath_kitty = "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/KITTI_2015/Common_corruptions/no_corruption/severity_0"
datapath_sceneflow = "/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/Common_corruptions/no_corruption/severity_0"

dataset_kitty = {
    'name' : 'kitti',
    'path': datapath_kitty
}

dataset_sceneflow = {
    'name' : 'sceneflow',
    'path' : datapath_sceneflow
}

# datasets = [dataset_kitty, dataset_sceneflow]
modes = ['train', 'validation', 'test', 'corrupted']
architectures = ['cfnet', 'gwcnet-g', 'sttr', 'sttr-light'] #'cfnet', 'gwcnet-g',

datasets = [dataset_sceneflow]
# modes = ['test', 'train', 'validation']
# architectures = ['cfnet']

for dataset in datasets:
    for architecture_name in architectures:
        for split in modes:
            data = get_dataset(
                        dataset['name'], dataset['path'], architecture_name, split, debug=True, random_seed=42)
            print(f"{dataset['name']} - {architecture_name} - {split}: Dataset ({len(data)})")
            assert len(data) == 10
            for x in data:
                assert x is not None
            print(f"{dataset['name']} - {architecture_name} - {split}: Elements are not None")
            loader = DataLoader(
                data,
                5,
                shuffle=False,
                num_workers=8,
                drop_last=True,
            )
            for x in loader:
                assert x is not None
            print(f"{dataset['name']} - {architecture_name} - {split}: Batches are not None")
        print(f"--- finished: {architecture_name} ---")

# # pudb.set_trace()
# indices = [0, 3, 5]  # example indices
# for idx in indices:
#     image = data[idx]
#     print(f"Index: {idx}, Image shape: {image}")
#
# # for x in data[0]:
# #     print(x)
#
# train_img_loader =
#
#
# for index, x in enumerate(train_img_loader):
#     print(index)
#     #print(x)