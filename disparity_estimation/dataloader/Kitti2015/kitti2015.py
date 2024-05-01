# sttr Kitti2015 datasetloader as base kittidataloader 
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations import Compose
from natsort import natsorted
import random
import re
import torchvision.transforms.functional as F
from dataloader.Kitti2015.sttr_preprocess import augment, normalization
from dataloader.Kitti2015.sttr_stereo_albumentation import RGBShiftStereo, RandomBrightnessContrastStereo, random_crop
from dataloader.Kitti2015.gwcnet_data_io import get_transform
import torch



class KITTIBaseDataset(data.Dataset):
    def __init__(self, datadir,model_name, split='train'):
        super(KITTIBaseDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        self.model_name = model_name

        if split == 'train' or split == 'validation' or split == 'validation_all':
            self.sub_folder = 'training/'
        elif split == 'test':
            self.sub_folder = 'testing/'

        self.left_fold = 'image_2/'
        self.right_fold = 'image_3/'
        self.disp_fold = 'disp_occ_0/'  # we read disp data with occlusion since we compute occ directly

        self._read_data()
        self._augmentation()

    def _read_data(self):
        assert self.left_fold is not None

        self.left_data = natsorted([os.path.join(self.datadir, self.sub_folder, self.left_fold, img) for img in
                                    os.listdir(os.path.join(self.datadir, self.sub_folder, self.left_fold)) if
                                    img.find('_10') > -1])
        self.right_data = [img.replace(self.left_fold, self.right_fold) for img in self.left_data]
        self.disp_data = [img.replace(self.left_fold, self.disp_fold) for img in self.left_data]

        self._split_data()

    def _split_data(self):
        train_val_frac = 0.95
        # split data
        if len(self.left_data) > 1:
            if self.split == 'train':
                self.left_data = self.left_data[:int(len(self.left_data) * train_val_frac)]
                self.right_data = self.right_data[:int(len(self.right_data) * train_val_frac)]
                self.disp_data = self.disp_data[:int(len(self.disp_data) * train_val_frac)]
            elif self.split == 'validation':
                self.left_data = self.left_data[int(len(self.left_data) * train_val_frac):]
                self.right_data = self.right_data[int(len(self.right_data) * train_val_frac):]
                self.disp_data = self.disp_data[int(len(self.disp_data) * train_val_frac):]

    def _augmentation(self):
        if self.model_name == 'STTR':
            if self.split == 'train':
                self.transformation = Compose([
                    RGBShiftStereo(always_apply=True, p_asym=0.5),
                    RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
                ])
            elif self.split == 'validation' or self.split == 'test' or self.split == 'validation_all':
                self.transformation = None
            else:
                raise Exception("Split not recognized")
        else:
            self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        if self.model_name == 'STTR':
            return self.get_item_STTR(idx)
        
        elif self.model_name == 'gwcnet-g':
            return self.get_item_GWCNET(idx)

        elif self.model_name == 'CFNet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")

        elif self.model_name == 'HSMNet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented") 
        
        elif self.model_name == 'PSMNet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")

        else:
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")

   


    def get_item_STTR(self,idx):
        input_data = {}

        # left
        left_fname = self.left_data[idx]
        left = np.array(Image.open(left_fname)).astype(np.uint8)
        input_data['left'] = left

        # right
        right_fname = self.right_data[idx]
        right = np.array(Image.open(right_fname)).astype(np.uint8)
        input_data['right'] = right

        # disp
        if not self.split == 'test':  # no disp for test files
            disp_fname = self.disp_data[idx]

            disp = np.array(Image.open(disp_fname)).astype(float) / 256.
            input_data['disp'] = disp
            input_data['occ_mask'] = np.zeros_like(disp).astype(bool)

            if self.split == 'train':
                input_data = random_crop(200, 640, input_data, self.split)

            input_data = augment(input_data, self.transformation)
        else:
            input_data = normalization(**input_data)

        return input_data
        

    
    def get_item_GWCNET(self, idx):
        left_img_path = os.path.join(self.datadir, self.sub_folder, self.left_fold, self.left_data[idx])
        right_img_path = os.path.join(self.datadir, self.sub_folder, self.right_fold, self.right_data[idx])

        left_img = Image.open(left_img_path).convert('RGB')
        # print(type(left_img))
        right_img = Image.open(right_img_path).convert('RGB')

        disp = None
        disp_img_path = os.path.join(self.datadir, self.sub_folder, self.disp_fold, self.disp_data[idx])
        disp = np.array(Image.open(disp_img_path)).astype(float) / 256.
        # if self.split != 'test':
            # disp_img_path = os.path.join(self.datadir, self.sub_folder, self.disp_fold, self.disp_data[idx])
            # disp = np.array(Image.open(disp_img_path)).astype(float) / 256.

        if self.split == 'train':
            return self.process_for_training(left_img, right_img, disp)
        else:
            return self.process_for_evaluation(left_img, right_img, disp)
        # left_img = self.load_image(os.path.join(self.datadir, self.left_data[idx]))
        # right_img = self.load_image(os.path.join(self.datadir, self.right_data[idx]))

        # if self.disp_data[idx]:  # has disparity ground truth
        #     disparity = self.load_disp(os.path.join(self.datadir, self.disp_data[idx]))
        # else:
        #     disparity = None

        # if self.split == 'train':
        #     return self.process_for_training(left_img, right_img, disparity)
        # else:
        #     return self.process_for_evaluation(left_img, right_img, disparity)

    
    def process_for_training(self, left_img, right_img, disparity):
        # Apply the same preprocessing steps as in data_io.py
        # transform = get_transform()
        # left_img = transform(left_img)
        # right_img = transform(right_img)

        # additional preprocessing steps specific for GWCNet training
        # print(type(left_img))
        w, h = left_img.size
        
        crop_w, crop_h = 512, 256

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

        # random crop
        left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

        transform = get_transform()
        left_img = transform(left_img)
        right_img = transform(right_img)


        # # to tensor, normalize
        # left_img = F.to_tensor(left_img)
        # right_img = F.to_tensor(right_img)

        return {"left": left_img,
                "right": right_img,
                "disparity": torch.from_numpy(disparity)}

    def process_for_evaluation(self, left_img, right_img, disparity):
        # Apply the same preprocessing steps as in data_io.py
        # transform = get_transform()
        # left_img = transform(left_img)
        # right_img = transform(right_img)



        # add additional preprocessing steps specific for GWCNet 
        w, h = left_img.size


        transform = get_transform()
        left_img = transform(left_img)
        right_img = transform(right_img)
        # normalize
        # left_img = F.to_tensor(left_img).numpy()
        # right_img = F.to_tensor(right_img).numpy()

        # pad to size 1248x384
        top_pad = 384 - h
        right_pad = 1248 - w
        assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = F.pad(left_img, (0, right_pad, top_pad, 0))
        right_img = F.pad(right_img, (0, right_pad, top_pad, 0))
        # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        # pad disparity gt
        if disparity is not None:
            assert len(disparity.shape) == 2
            disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        if disparity is not None:
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": top_pad,
                    "right_pad": right_pad}
        else:
            return {"left": left_img,
                    "right": right_img,
                    "top_pad": top_pad,
                    "right_pad": right_pad}