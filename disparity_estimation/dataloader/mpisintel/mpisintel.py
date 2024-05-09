import os

import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import torch
from PIL import Image
from natsort import natsorted

from ..preprocess import augment

def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:, :, 0].astype('float64')
    d_g = f_in[:, :, 1].astype('float64')
    d_b = f_in[:, :, 2].astype('float64')

    depth = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    return depth

class MPISintelDataset(data.Dataset):
    def __init__(self, datadir, model_name='psmnet'):
        super(MPISintelDataset, self).__init__()
        
        self.model_name = model_name.lower()
        self.datadir = datadir
        self._read_data()
        self._augmentation()

    def _read_data(self):
        directory = os.path.join(self.datadir, 'final_left')
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

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

            print(input_data_processed['disparity'].shape, input_data_processed['left'].shape, input_data_processed['right'].shape)
            print(type(input_data_processed['disparity']))
            # input_data_processed['disparity'] = input_data_processed['disparity'][:432, :]
            # input_data_processed['left'] = input_data_processed['left'][:, :432, :]
            # input_data_processed['right'] = input_data_processed['right'][:, :432, :]

            input_data_processed['disparity'] = torch.from_numpy(input_data_processed['disparity'])
            # input_data_processed['left'] = torch.from_numpy(input_data_processed['left'])
            # input_data_processed['right'] = torch.from_numpy(input_data_processed['right'])

            target_size = (3, 256, 512)
            input_data_processed['disparity'] = F.interpolate(input_data_processed['disparity'], size=(256, 512), mode='bilinear', align_corners=False)
            input_data_processed['left'] = F.interpolate(input_data_processed['left'], size=target_size, mode='bilinear', align_corners=False)
            input_data_processed['right'] = F.interpolate(input_data_processed['right'], size=target_size, mode='bilinear', align_corners=False)
            
            input_data_processed['disparity'] = input_data_processed['disparity'].numpy()
            # input_data_processed['left'] = input_data_processed['left'].numpy()
            # input_data_processed['right'] = input_data_processed['right'].numpy()

            # (436, 1024) torch.Size([3, 436, 1024]) torch.Size([3, 436, 1024])
            print('new: ',input_data_processed['disparity'].shape, input_data_processed['left'].shape, input_data_processed['right'].shape)
            
            
            return input_data_processed
        elif self.model_name == 'psmnet':
            input_data_processed = self.preprocess_item_STTR(input_data_raw)
            return (input_data_processed['left'], input_data_processed['right'], input_data_processed['disp'])
        
    
    def __load_element_from_disk__(self, idx):
        input_data = {}

        left_fname = self.left_data[idx]
        input_data['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]

        right_fname = left_fname.replace('final_left', 'final_right')
        input_data['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]

        disp_left_fname = left_fname.replace('final_left', 'disparities')
        disp_left = disparity_read(disp_left_fname)

        occ_left_fname = left_fname.replace('final_left', 'occlusions')
        occ_left_occ = np.array(Image.open(occ_left_fname))
        occ_left_fname = left_fname.replace('final_left', 'outofframe')
        occ_left_oof = np.array(Image.open(occ_left_fname))
        occ_left = np.logical_or(occ_left_occ, occ_left_oof)

        input_data['occ_mask'] = occ_left
        input_data['disp'] = disp_left

        return input_data

    def preprocess_item_STTR(self, input_data_raw):

        return augment(input_data_raw, self.transformation)

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