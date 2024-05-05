from PIL import Image
import numpy as np

def print_labels_by_image(image_str: str):
    image = Image.open(image_str)

    data = np.asarray(image)

    print(np.unique(data.flatten()))

folder = "mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClassAug"
print_labels_by_image(folder + "/2007_000032.png")
print_labels_by_image(folder + "/2007_000033.png")
print_labels_by_image(folder + "/2007_000039.png")
print_labels_by_image(folder + "/2007_000042.png")
