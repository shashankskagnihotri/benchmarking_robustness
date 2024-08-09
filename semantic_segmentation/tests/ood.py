from PIL import Image
import numpy as np
from imagenet_c import corrupt

# ade
# fpath = "/pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/mmsegmentation/data/ade/ADEChallengeData2016/images/training/ADE_train_00000002.jpg"

# cityscapes
# fpath = "/pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/mmsegmentation/data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"

# VOC
fpath = "/pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/mmsegmentation/data/VOCdevkit/VOCaug/dataset/img/2008_000008.jpg"


folder = "/pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/ood_test_images"

img = Image.open(fpath)
img = np.array(img)
print(img.shape)

im = Image.fromarray(img)
im.save(folder + f"/img.png")


for i in range(19):
    # frost

    if i == 8:
        continue

    c = corrupt(img, corruption_number = i)

    im = Image.fromarray(c)
    im.save(folder + f"/c{i}.png")
