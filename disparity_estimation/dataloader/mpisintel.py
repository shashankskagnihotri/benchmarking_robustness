import os

import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import torch
from PIL import Image
from natsort import natsorted

# Imports
from . import cfnet, sttr, sttr_light, psmnet, hsmnet, gwcnet


def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:, :, 0].astype('float64')
    d_g = f_in[:, :, 1].astype('float64')
    d_b = f_in[:, :, 2].astype('float64')

    depth = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    return depth

def convert_items_to_tensor(input_data_raw):
    input_data_processed = {}
    for key, value in input_data_raw.items():
        input_data_processed[key] = torch.from_numpy(value)
    return input_data_processed

def convert_items_to_numpy(input_data_raw):
    input_data_processed = {}
    for key, value in input_data_raw.items():
        input_data_processed[key] = value.numpy()
    return input_data_processed

def convert_items_from_pil_image_to_numpy(input_data_raw):
    input_data_processed = {}
    for key, value in input_data_raw.items():
        input_data_processed[key] = np.array(value)
    return input_data_processed

class MPISintelDataset(data.Dataset):
    def __init__(self, datadir:str, model_name:str, train:str):
        super(MPISintelDataset, self).__init__()
        
        self.model_name = model_name.lower()
        self.datadir = datadir
        self.training = True if train.lower() == "train" else False
        self.split = train
        
        self._read_data()
        self._augmentation()

    def _read_data(self):
        if not os.path.isdir(self.datadir):
            raise ValueError(f"Could not find the directory {self.datadir}")
        
        directory = os.path.join(self.datadir, "train", 'final_left')
        print(directory, os.path.isdir(directory))
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))] if os.path.isdir(directory) else []
        print("sub_folders: ", sub_folders[:3])
        self.left_data = []
        for sub_folder in sub_folders:
            self.left_data += [os.path.join(sub_folder, img) for img in
                               os.listdir(os.path.join(sub_folder))] 

        self.left_data = natsorted(self.left_data)


        
    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        # input_data_raw dict keys: left, right, occ_mask (left), disp (disp_left)
        input_data_raw = self.__load_element_from_disk__(idx) 


        if self.model_name == 'sttr':
            return self.preprocess_item_STTR(input_data_raw)
        elif self.model_name == 'sttr-light':
            return self.preprocess_item_STTR(input_data_raw)
        elif self.model_name == 'cfnet':
            input_data_processed = self.preprocess_item_STTR(input_data_raw)
            input_data_processed['disparity'] = input_data_processed.pop('disp')

            input_data_processed['disparity'] = input_data_processed['disparity'].squeeze() # here dimension is ([436, 1024]) numpay array
            input_data_processed['left'] = input_data_processed['left'].squeeze() # here dimension is ([3, 436, 1024]) numpay array
            input_data_processed['right'] = input_data_processed['right'].squeeze() # here dimension is ([3, 436, 1024]) numpay array

            random_crop_cf_net = cfnet.flow_transforms.RandomCropMPISintel((256, 512))
            input_data_processed['left'], input_data_processed['right'], input_data_processed['disparity'] = random_crop_cf_net(input_data_processed['left'], input_data_processed['right'], input_data_processed['disparity'])

            return input_data_processed
        
        elif self.model_name == 'psmnet':
            input_data_processed = self.preprocess_item_STTR(input_data_raw)
            return (input_data_processed['left'], input_data_processed['right'], input_data_processed['disp'])
    

    def generate_disparity_path(self, original_path:str) -> str:

        # Zerlege den originalen Pfad in seine Teile
        parts = original_path.split('/')

        # Finde den Index des Verzeichnisses 'FlyingThings3D'
        try:
            mpi_sintel_index = parts.index('mpisintel')
        except ValueError:
            raise ValueError("Der Pfad enthält kein 'mpisintel'-Verzeichnis.")

        
        # Ersetze den Pfad ab 'FlyingThings3D' mit dem neuen Pfad
        parts[mpi_sintel_index+2] = "no_corruption"
        parts[mpi_sintel_index+3] = "severity_0"
        parts[mpi_sintel_index+5] = "disparities"

        # Erstelle den neuen Pfad
        new_path = "/" + os.path.join(*parts)
        
        return new_path
    
    def __load_element_from_disk__(self, idx):
        input_data = {}

        left_fname = self.left_data[idx]
        input_data['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]

        right_fname = left_fname.replace('final_left', 'final_right')
        input_data['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]
        disp_left_fname = self.generate_disparity_path(left_fname)
        disp_left = disparity_read(disp_left_fname)

        occ_left_fname = disp_left_fname.replace('disparities', 'occlusions')
        occ_left_occ = np.array(Image.open(occ_left_fname))
        occ_left_fname = disp_left_fname.replace('disparities', 'outofframe')
        occ_left_oof = np.array(Image.open(occ_left_fname))
        occ_left = np.logical_or(occ_left_occ, occ_left_oof)

        input_data['occ_mask'] = occ_left
        input_data['disp'] = disp_left

        return input_data

    def preprocess_item_STTR(self, input_data_raw):

        return sttr.preprocess.augment(input_data_raw, self.transformation)

# ---------------------------------------------------------------------------- FROM STTR ---------------------------------------------------------------------------- #
#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2021. Johns Hopkins University - All rights reserved.

# import os

# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# from natsort import natsorted

# from dataset.preprocess import augment


# def disparity_write(filename, disparity, bitdepth=16):
#     """ Write disparity to file.

#     bitdepth can be either 16 (default) or 32.

#     The maximum disparity is 1024, since the image width in Sintel
#     is 1024.
#     """
#     d = disparity.copy()

#     # Clip disparity.
#     d[d > 1024] = 1024
#     d[d < 0] = 0

#     d_r = (d / 4.0).astype('uint8')
#     d_g = ((d * (2.0 ** 6)) % 256).astype('uint8')

#     out = np.zeros((d.shape[0], d.shape[1], 3), dtype='uint8')
#     out[:, :, 0] = d_r
#     out[:, :, 1] = d_g

#     if bitdepth > 16:
#         d_b = (d * (2 ** 14) % 256).astype('uint8')
#         out[:, :, 2] = d_b

#     Image.fromarray(out, 'RGB').save(filename, 'PNG')


# def disparity_read(filename):
#     """ Return disparity read from filename. """
#     f_in = np.array(Image.open(filename))
#     d_r = f_in[:, :, 0].astype('float64')
#     d_g = f_in[:, :, 1].astype('float64')
#     d_b = f_in[:, :, 2].astype('float64')

#     depth = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
#     return depth


# class SintelDataset(data.Dataset):
#     def __init__(self, datadir, split='train'):
#         super(SintelDataset, self).__init__()

#         self.datadir = datadir
#         self.split = split # this variable is not used 
#         self._read_data()
#         self._augmentation()

#     def _read_data(self):
#         directory = os.path.join(self.datadir, 'final_left')
#         sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
#                        os.path.isdir(os.path.join(directory, subset))]

#         self.left_data = []
#         for sub_folder in sub_folders:
#             self.left_data += [os.path.join(sub_folder, img) for img in
#                                os.listdir(os.path.join(sub_folder))]

#         self.left_data = natsorted(self.left_data)

#     def _augmentation(self):
#         self.transformation = None

#     def __len__(self):
#         return len(self.left_data)

#     def __getitem__(self, idx):
#         input_data = {}

#         left_fname = self.left_data[idx]
#         input_data['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]

#         right_fname = left_fname.replace('final_left', 'final_right')
#         input_data['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]

#         disp_left_fname = left_fname.replace('final_left', 'disparities')
#         disp_left = disparity_read(disp_left_fname)

#         occ_left_fname = left_fname.replace('final_left', 'occlusions')
#         occ_left_occ = np.array(Image.open(occ_left_fname))
#         occ_left_fname = left_fname.replace('final_left', 'outofframe')
#         occ_left_oof = np.array(Image.open(occ_left_fname))
#         occ_left = np.logical_or(occ_left_occ, occ_left_oof)

#         input_data['occ_mask'] = occ_left
#         input_data['disp'] = disp_left

#         input_data = augment(input_data, self.transformation)

#         return input_data
# ---------------------------------------------------------------------------- FROM STTR (END) ---------------------------------------------------------------------------- #