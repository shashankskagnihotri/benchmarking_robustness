import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
import copy
from albumentations import Compose, OneOf
from natsort import natsorted

# Imports
from . import cfnet, sttr, sttr_light, psmnet, hsmnet, gwcnet


# SceneFlow dataloader from CFNet
class SceneFlowFlyingThings3DDataset(Dataset):
    def __init__(self, datadir, architecture_name, split='train'):
        super().__init__()

        self.datadir = datadir
        self.model_name = architecture_name.lower()
        
        if split.upper() == 'TRAIN':
            self.split_folder = 'TRAIN'
        elif split.upper() == 'TEST':
            self.split_folder = 'TEST'
        elif split.upper() == 'CORRUPTED':
            self.split_folder = 'CORRUPTED'
        else:
            raise ValueError(f"Invalid split value: {split}")

        self.training = True if split.lower() in "train" else False

        self._read_data()

    def _read_data(self):

        import os

        def generate_disparity_path(original_path:str) -> str:
            # Zerlege den originalen Pfad in seine Teile
            parts = original_path.split('/')

            # Finde den Index des Verzeichnisses 'FlyingThings3D'
            try:
                flyingthings3d_index = parts.index('FlyingThings3D')
            except ValueError:
                raise ValueError("Der Pfad enthält kein 'FlyingThings3D'-Verzeichnis.")

            # Ersetze den Pfad ab 'FlyingThings3D' mit dem neuen Pfad
            new_parts = parts[:flyingthings3d_index + 1] + ['disparity'] + parts[flyingthings3d_index + 5:]

            # Erstelle den neuen Pfad
            new_path = "/" + os.path.join(*new_parts)
            return new_path

        def generate_occlusion_path() -> str:

            # os.path.join(self.datadir, 'occlusion', self.split_folder, 'left')

            # Zerlege den originalen Pfad in seine Teile
            parts = self.datadir.split('/')
            
            # Finde den Index des Verzeichnisses 'FlyingThings3D'
            try:
                flyingthings3d_index = parts.index('FlyingThings3D')
            except ValueError:
                raise ValueError(f"Der Pfad enthält kein 'FlyingThings3D'-Verzeichnis: {self.datadir}")

            # Ersetze den Pfad ab 'FlyingThings3D' mit dem neuen Pfad
            new_parts = parts[:flyingthings3d_index + 1] + ['Common_corruptions'] + ['no_corruption'] + ['severity_0'] + ['frames_finalpass'] + ['occlusion'] + [self.split_folder] + ['left']

            # Erstelle den neuen Pfad
            new_path = "/" + os.path.join(*new_parts)
            return new_path

        

        directory = os.path.join(self.datadir, 'frames_finalpass', self.split_folder)
        print("Split-Folder: ", self.split_folder)
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))] if os.path.isdir(directory) else []

        seq_folders = []
        for sub_folder in sub_folders:
            seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                            os.path.isdir(os.path.join(sub_folder, seq))]

        self.img_left_filenames = []
        for seq_folder in seq_folders:
            self.img_left_filenames += [os.path.join(seq_folder, 'left', img) for img in
                               os.listdir(os.path.join(seq_folder, 'left'))]

        # Remove unused files (in sttr)
        if self.model_name == 'sttr':
            flyingthings3d_index = self.datadir.split('/').index('FlyingThings3D')
            path_unused_files = os.path.join('/', *(self.datadir.split('/')[:flyingthings3d_index + 1] + ['all_unused_files.txt']))
            unused_files = [line.strip().rstrip() for line in open(path_unused_files, mode='r').read().splitlines()]
            
            number_of_images_before = len(self.img_left_filenames)
            new_img_left_filenames = []
            for img in self.img_left_filenames:
                add_image = True  # Flag, um festzuhalten, ob das Bild hinzugefügt werden soll
                for unused in unused_files:
                    if unused in img:  # Wenn ein ungenutztes Bild gefunden wird
                        add_image = False  # Setze Flag auf False
                        break  # Breche die Schleife ab, da das Bild nicht hinzugefügt werden soll
                if add_image:  # Wenn das Bild nicht in unused_files ist
                    # print(img)  # Optional: Zum Debuggen, drucke das Bild aus
                    new_img_left_filenames.append(img)  # Füge das Bild zur Liste hinzu

                
            self.img_left_filenames = new_img_left_filenames
            number_of_images_after = len(self.img_left_filenames)

            print(f"Inital number of images: {number_of_images_before}")
            print(f"Removed {number_of_images_before - number_of_images_after} unused files")
        
        

        self.img_left_filenames = natsorted(self.img_left_filenames)
        self.img_right_filenames = [img_path.replace('left', 'right') for img_path in self.img_left_filenames]
        
        self.disp_left_filenames = [generate_disparity_path(img_path).replace('.png', '.pfm') for img_path in self.img_left_filenames]
        self.disp_right_filenames = [generate_disparity_path(img_path).replace('.png', '.pfm') for img_path in self.img_right_filenames]

        directory = generate_occlusion_path()
        self.occ_left_filenames = [os.path.join(directory, occ) for occ in os.listdir(directory)] if os.path.isdir(directory) else []
        self.occ_left_filenames = natsorted(self.occ_left_filenames)
        self.occ_right_filenames = [img_path.replace('left', 'right') for img_path in self.occ_left_filenames]

        print("Final number of images: ", number_of_images_after)
        print("Final number of occlusion files: ", len(self.occ_left_filenames))
        print("Final number of disparity files: ", len(self.disp_left_filenames))
        print("")
        

    def load_image(self, filename) -> Image:
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename) -> np.ndarray[np.float32]:
        return cfnet.data_io.pfm_imread(filename)[0].astype(np.float32)
    
    def load_occ(self, filename) -> np.ndarray[bool]:
        return np.array(Image.open(filename)).astype(bool)
    

    def __len__(self) -> int:
        return len(self.img_left_filenames)

    
    def __getitem__(self, index):
        img_left = self.load_image(self.img_left_filenames[index])
        img_right = self.load_image(self.img_right_filenames[index])
        disp_left = self.load_disp(self.disp_left_filenames[index])
        disp_right = self.load_disp(self.disp_right_filenames[index])
        
        if self.model_name == 'cfnet':
            return self.get_item_cfnet(img_left, img_right, disp_left)
        elif self.model_name == 'psmnet':
            return self.get_item_psmnet(img_left, img_right, disp_left)
        elif self.model_name in ['gwcnet', 'gwcnet-g', 'gwcnet-gc']:
            return self.get_item_gwcnet(img_left, img_right, disp_left)
        elif self.model_name == 'sttr':
            # This might give an error here because there are less oclusion files then others
            occ_left = self.load_occ(self.occ_left_filenames[index])
            occ_right = self.load_occ(self.occ_right_filenames[index])
            return self.get_item_sttr(img_left, img_right, disp_left, disp_right, occ_left, occ_right)
        elif self.model_name == 'hsmnet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        
        else:
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")




    














    def get_item_sttr(self, left_img:Image, right_img:Image, left_disp:np.ndarray[np.float32], right_disp:np.ndarray[np.float32], left_occ:np.ndarray[bool], right_occ:np.ndarray[bool]) -> dict:
        result = {}

        # left_fname = self.left_data[idx]
        # result['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]
        result['left'] = np.array(left_img).astype(np.uint8)

        # right_fname = left_fname.replace('left', 'right')
        # result['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]
        result['right'] = np.array(right_img).astype(np.uint8)

        # occ_right_fname = self.occ_data[idx].replace('left', 'right')
        # occ_left = np.array(Image.open(self.occ_data[idx])).astype(bool)
        # occ_right = np.array(Image.open(occ_right_fname)).astype(bool)

        # disp_left_fname = left_fname.replace('frames_finalpass', 'disparity').replace('.png', '.pfm')
        # disp_right_fname = right_fname.replace('frames_finalpass', 'disparity').replace('.png', '.pfm')
        # disp_left, _ = readPFM(disp_left_fname)
        # disp_right, _ = readPFM(disp_right_fname)



        if self.training:
            # horizontal flip
            result['left'], result['right'], result['occ_mask'], result['occ_mask_right'], disp, disp_right \
                = sttr.stereo_albumentation.horizontal_flip(result['left'], result['right'], left_occ, right_occ, left_disp, disp_right,
                                  self.split_folder)
            result['disp'] = np.nan_to_num(disp, nan=0.0)
            result['disp_right'] = np.nan_to_num(disp_right, nan=0.0)

            # random crop        
            result = sttr.stereo_albumentation.random_crop(360, 640, result, self.split_folder)
        else:
            result['occ_mask'] = left_occ
            result['occ_mask_right'] = right_occ
            result['disp'] = left_disp
            result['disp_right'] = right_disp

        result = sttr.preprocess.augment(result, Compose([
                sttr.stereo_albumentation.RandomShiftRotate(always_apply=True),
                sttr.stereo_albumentation.RGBShiftStereo(always_apply=True, p_asym=0.3),
                OneOf([
                    sttr.stereo_albumentation.GaussNoiseStereo(always_apply=True, p_asym=1.0, p=False),
                    sttr.stereo_albumentation.RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
                ], p=1.0)
            ]))

        return result



    def get_item_psmnet(self, left_img:Image, right_img:Image, disparity:Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        left_img = np.ascontiguousarray(np.array(left_img))
        right_img = np.ascontiguousarray(np.array(right_img))
        disparity = np.ascontiguousarray(np.array(disparity)).astype(np.float32)
        
        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            disparity = disparity[y1:y1 + th, x1:x1 + tw]

            processed = psmnet.preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img, disparity
        else:
            processed = psmnet.preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img) 
            return left_img, right_img, disparity



    def get_item_cfnet(self, left_img, right_img, disparity) -> dict:
        
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
            right_img = np.array(right_img)
            left_img = np.array(left_img)

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
            co_transform = cfnet.flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                cfnet.flow_transforms.RandomCrop((th, tw)),
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
            processed = cfnet.data_io.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)



            return {"left": torch.Tensor(left_img),
                    "right": torch.Tensor(right_img),
                    "disparity": torch.Tensor(disparity)}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = cfnet.data_io.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": torch.Tensor(left_img),
                    "right": torch.Tensor(right_img),
                    "disparity": torch.Tensor(disparity),
                    "top_pad": 0,
                    "right_pad": 0}







    def get_item_gwcnet(self, left_img:Image, right_img:Image, disparity:Image) -> dict:
        
        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            left_img = np.ascontiguousarray(np.array(left_img))
            right_img = np.ascontiguousarray(np.array(right_img))
            disparity = np.ascontiguousarray(np.array(disparity)).astype(np.float32)


            # to tensor, normalize
            processed = gwcnet.data_io.get_transform()
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

            processed = gwcnet.data_io.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0}