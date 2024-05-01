import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision
import torch
import torchvision.transforms as transforms
import cv2
import copy




from . import CFNet_flow_transforms as flow_transforms
from .sttr_stereo_albumentation import random_crop, horizontal_flip
from .sttr_preprocess import augment
from natsort import natsorted


# SceneFlow dataloader from CFNet
class SceneFlowFlyingThings3DDataloader(Dataset):
    def __init__(self, datadir, split='train'):
        super(SceneFlowFlyingThings3DDataloader, self).__init__()

        self.datadir = datadir
        self.split = split

        if self.split == 'train':
            self.split_folder = 'TRAIN'
        else:
            self.split_folder = 'TEST'

        self._read_data()
        self._augmentation()

    def _read_data(self):
        directory = os.path.join(self.datadir, 'frames_finalpass', self.split_folder)
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

        seq_folders = []
        for sub_folder in sub_folders:
            seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                            os.path.isdir(os.path.join(sub_folder, seq))]

        self.left_data = []
        for seq_folder in seq_folders:
            self.left_data += [os.path.join(seq_folder, 'left', img) for img in
                               os.listdir(os.path.join(seq_folder, 'left'))]

        self.left_data = natsorted(self.left_data)

        directory = os.path.join(self.datadir, 'occlusion', self.split_folder, 'left')
        self.occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        self.occ_data = natsorted(self.occ_data)

    

    def __len__(self):
        return len(self.left_data)

    
    def __getitem__(self, index):
        img_left = self.load_image(os.path.join(self.datapath, self.img_left_filenames[index]))
        img_right = self.load_image(os.path.join(self.datapath, self.img_right_filenames[index]))
        disp_left = self.load_disp(os.path.join(self.datapath, self.disp_left_filenames[index]))
        disp_right = self.load_disp(os.path.join(self.datapath, self.disp_right_filenames[index]))

        if self.model_name == 'CFNet':
            return self.getitem_CFNet(img_left, img_right, disp_left)
        elif self.model_name == 'PSMNet':
            return self.get_item_PSMNet(img_left, img_right, disp_left, disp_right)
        elif self.model_name == 'GWCNet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        elif self.model_name == 'HSMNet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        elif self.model_name == 'STTR':
            return self.item_
        
        else:
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")




    














    def get_item_STTR(self, left_img, right_img, left_disp, right_disp, left_occ, right_occ) -> dict:
        result = {}

        # left_fname = self.left_data[idx]
        # result['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]
        result['left'] = left_img

        # right_fname = left_fname.replace('left', 'right')
        # result['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]
        result['right'] = right_img

        # occ_right_fname = self.occ_data[idx].replace('left', 'right')
        # occ_left = np.array(Image.open(self.occ_data[idx])).astype(bool)
        # occ_right = np.array(Image.open(occ_right_fname)).astype(bool)

        # disp_left_fname = left_fname.replace('frames_finalpass', 'disparity').replace('.png', '.pfm')
        # disp_right_fname = right_fname.replace('frames_finalpass', 'disparity').replace('.png', '.pfm')
        # disp_left, _ = readPFM(disp_left_fname)
        # disp_right, _ = readPFM(disp_right_fname)



        if self.split == "train":
            # horizontal flip
            result['left'], result['right'], result['occ_mask'], result['occ_mask_right'], disp, disp_right \
                = horizontal_flip(result['left'], result['right'], left_occ, right_occ, left_disp, disp_right,
                                  self.split)
            result['disp'] = np.nan_to_num(disp, nan=0.0)
            result['disp_right'] = np.nan_to_num(disp_right, nan=0.0)

            # random crop        
            result = random_crop(360, 640, result, self.split)
        else:
            result['occ_mask'] = left_occ
            result['occ_mask_right'] = right_occ
            result['disp'] = left_disp
            result['disp_right'] = right_disp

        result = augment(result, self.transformation)

        return result



    def get_item_PSMNet(self, left_img, right_img, disparity) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        def get_transform():
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img, dataL
        else:
            processed = get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img) 
            return left_img, right_img, dataL



    def get_item_CFNet(self, left_img, right_img, disparity) -> dict:
        
        if self.training:

            th, tw = 256, 512
            #th, tw = 288, 512
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
            angle = 0
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

            # randomly occlude a region
            right_img.flags.writeable = True
            if np.random.binomial(1,0.5):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # w, h = left_img.size

            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)



            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0}
