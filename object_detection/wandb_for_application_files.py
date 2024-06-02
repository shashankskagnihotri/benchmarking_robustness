import os
from mmengine.config import Config

folder_path = "configs_verified_voc"

folder = os.listdir(folder_path)

def name_finder(file):
    name = file.split("_")
    neck = name[0]
    backbone = name[1]
    dataset = name[2]
    return neck, backbone, dataset

# join paths


for file in folder:
    cfg = Config.fromfile(os.path.join(folder_path, file))
    neck, backbone, dataset = name_finder(file)

    cfg.visualizer.vis_backends[0].type = "WandbVisBackend"
    cfg.visualizer.vis_backends[0].init_kwargs = dict(
    project=f"{neck}_{backbone}_{dataset}")

    cfg.dump(os.path.join(folder_path, file))