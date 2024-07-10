
# sttr Kitti2015 datasetloader as base kittidataloader 
import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
from albumentations import Compose
from natsort import natsorted
import random
import re
import torchvision.transforms.functional as F
import torch
from torchvision import transforms


# Imports
from . import cfnet, sttr, sttr_light, psmnet, hsmnet, gwcnet
from .sttr.stereo_albumentation import RGBShiftStereo, RandomBrightnessContrastStereo, random_crop


# from dataloader.kitti2015.sttr_preprocess import augment, normalization
# from dataloader.kitti2015.sttr_stereo_albumentation import RGBShiftStereo, RandomBrightnessContrastStereo, random_crop
# from dataloader.kitti2015.gwcnet_data_io import get_transform



# from .kitti2015 import psmnet_preprocess 
# from .cfnet_data_io import get_transform_cfnet, pfm_imread_cfnet

# test

class KITTIBaseDataset(data.Dataset):
    def __init__(self, datadir, architecture_name, split='train'):
        # ['train', 'test', 'validation', 'validation_all', 'corrupted']
        super(KITTIBaseDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        self.architecture_name = architecture_name.lower()

        if split == 'train' or split == 'validation' or split == 'validation_all':
            self.sub_folder = 'training/'
        elif split == 'test':
            self.sub_folder = 'testing/'

        self.left_fold = 'image_2/'
        self.right_fold = 'image_3/'
        self.disp_fold = 'disp_occ_0/' 
        # if split=='train':
        #     self.disp_fold = 'disp_occ_0/'  # we read disp data with occlusion since we compute occ directly
        # else:
        #     self.disp_fold = None


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
        if self.architecture_name == 'sttr' or self.architecture_name == 'sttr-light':
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
    

    def load_image(self, filename) -> Image:
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename) -> np.ndarray[np.float32]:
        return np.array(Image.open(filename), dtype=np.float32) / 256.
        
    def load_occ(self, filename) -> np.ndarray[bool]:
        return np.array(Image.open(filename)).astype(bool)
    

    def __getitem__(self, index):
        

        img_left = self.load_image(self.left_data[index])
        img_right = self.load_image(self.right_data[index])
        disp_left = self.load_disp(self.disp_data[index])
        
        
        
        if self.architecture_name == 'sttr':
            return self.get_item_STTR(img_left, img_right, disp_left)
        
        elif self.architecture_name == 'sttr-light': 
            return self.get_item_STTR(img_left, img_right, disp_left)
        
        elif self.architecture_name == 'gwcnet-g' or self.architecture_name == 'gwcnet-gc':
            return self.get_item_GWCNET(img_left, img_right, disp_left)

        elif self.architecture_name == 'cfnet':
            raise NotImplemented(f"No dataloder for {self.architecture_name} implemented")

        elif self.architecture_name == 'hsmnet':
            raise NotImplemented(f"No dataloder for {self.architecture_name} implemented") 
        
        elif self.architecture_name == 'psmnet':
            return self.get_item_PSMNet(img_left, img_right, disp_left)
        else:
            raise NotImplemented(f"No dataloder for {self.architecture_name} implemented")

   
    def get_item_PSMNet(self, left_img, right_img, dataL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = psmnet.preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1232, h-368, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           processed = psmnet.preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL


    def get_item_STTR(self, img_left:Image, img_right:Image, disp_left:np.ndarray) -> dict:
        input_data = {}

        # left
        # left_fname = self.left_data[idx]
        # left = np.array(Image.open(left_fname)).astype(np.uint8)
        # input_data['left'] = left
        input_data['left'] = np.array(img_left).astype(np.uint8)

        # right
        # right_fname = self.right_data[idx]
        # right = np.array(Image.open(right_fname)).astype(np.uint8)
        # input_data['right'] = right
        input_data['right'] = np.array(img_right).astype(np.uint8)

        # disp
        if not self.split == 'test':  # no disp for test files
            # disp_fname = self.disp_data[idx]

            # disp = np.array(Image.open(disp_fname)).astype(float) / 256.
            input_data['disp'] = np.array(disp_left).astype(float) / 256.
            input_data['occ_mask'] = np.zeros_like(disp_left).astype(bool)

            if self.split == 'train':
                input_data = random_crop(200, 640, input_data, self.split)

            input_data = sttr.preprocess.augment(input_data, self.transformation)
        else:
            input_data = sttr.preprocess.normalization(**input_data)

        return input_data
    
    def get_item_sttr_light(self, img_left:Image, img_right:Image, disp_left:np.ndarray) -> dict:
        input_data = {}

        # left
        # left_fname = self.left_data[idx]
        # left = np.array(Image.open(left_fname)).astype(np.uint8)
        input_data['left'] = np.array(img_left).astype(np.uint8)

        # right
        # right_fname = self.right_data[idx]
        # right = np.array(Image.open(right_fname)).astype(np.uint8)
        input_data['right'] = np.array(img_right).astype(np.uint8)

        # disp
        if not self.split == 'test':  # no disp for test files
            # disp_fname = self.disp_data[idx]

            # disp = np.array(Image.open(disp_fname)).astype(float) / 256.
            input_data['disp'] = np.array(disp_left).astype(float) / 256.
            input_data['occ_mask'] = np.zeros_like(disp_left).astype(bool)

            if self.split == 'train':
                input_data = random_crop(200, 640, input_data, self.split)

            input_data = sttr.preprocess.augment(input_data, self.transformation)
        else:
            input_data = sttr.preprocess.normalization(**input_data)

        return input_data
        

    
    def get_item_GWCNET(self, left_img:Image, right_img:Image, disparity:np.ndarray) -> dict:
        # left_img_path = os.path.join(self.datadir, self.sub_folder, self.left_fold, self.left_data[idx])
        # right_img_path = os.path.join(self.datadir, self.sub_folder, self.right_fold, self.right_data[idx])

        # left_img = Image.open(left_img_path).convert('RGB')
        # # print(type(left_img))
        # right_img = Image.open(right_img_path).convert('RGB')

        # disp = None
        # disp_img_path = os.path.join(self.datadir, self.sub_folder, self.disp_fold, self.disp_data[idx])
        # disp = np.array(Image.open(disp_img_path)).astype(float) / 256.
        
        
        
        # if self.split != 'test':
            # disp_img_path = os.path.join(self.datadir, self.sub_folder, self.disp_fold, self.disp_data[idx])
            # disp = np.array(Image.open(disp_img_path)).astype(float) / 256.

        if self.split == 'train':
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

            transform = gwcnet.data_io.get_transform()
            left_img = transform(left_img)
            right_img = transform(right_img)


            # # to tensor, normalize
            # left_img = F.to_tensor(left_img)
            # right_img = F.to_tensor(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": torch.from_numpy(disparity)}
        
        
        else:
            # Apply the same preprocessing steps as in data_io.py
            transform = transforms.Compose([
                transforms.Resize((384, 1248)),  # Resize to the target size
                transforms.ToTensor(),           # Convert PIL image to tensor
            ])

            left_img = transform(left_img)
            right_img = transform(right_img)

            # Pad disparity gt if not None
            if disparity is not None:
                # Resize the disparity map
                disparity = Image.fromarray(disparity)
                disparity = disparity.resize((1248, 384), Image.NEAREST)
                disparity = np.array(disparity, dtype=np.float32) / 256.

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity}
            else:
                return {"left": left_img,
                        "right": right_img}

            # if disparity is not None:
            #     return {"left": left_img,
            #             "right": right_img,
            #             "disparity": disparity,
            #             "top_pad": pad_height,
            #             "right_pad": pad_width}
            # else:
            #     return {"left": left_img,
            #             "right": right_img,
            #             "top_pad": pad_height,
            #             "right_pad": pad_width}
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


    def get_item_CFNET(self, left_img:Image, right_img:Image, disparity:np.ndarray) -> dict:
        if self.training:
            th, tw = 256, 512
            #th, tw = 320, 704
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            right_img = np.asarray(right_img)
            left_img = np.asarray(left_img)

            # w, h  = left_img.size
            # th, tw = 256, 512
            #
            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)
            #
            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # right_img = np.asarray(right_img)
            # left_img = np.asarray(left_img)

            # geometric unsymmetric-augmentation
            angle = 0;
            px = 0
            if np.random.binomial(1, 0.5):
                # angle = 0.1;
                # px = 2
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, disparity = co_transform([left_img, right_img], disparity)
            left_img = augmented[0]
            right_img = augmented[1]

            right_img.flags.writeable = True
            if np.random.binomial(1,0.2):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            processed = cfnet.data_io.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size

            # normalize
            processed = cfnet.data_io.get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
