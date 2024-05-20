import os, torch, torch.utils.data as data
from PIL import Image
import numpy as np
import pdb
import glob
import torchvision
import warnings

# Imports
from . import cfnet, sttr, sttr_light, psmnet, hsmnet, gwcnet


# ETH3D dataloader from CFNET
class ETH3DDataset(data.Dataset):


    def __init__(self, datadir:str, model_name:str, train:bool=True):
        super().__init__()

        self.datadir = datadir
        self.model_name = model_name.lower()

        self.split_folder = 'training' if train else 'test'
        self.training = train

        self._read_data()
        

    def _read_data(self):
        img_list = [i.split('/')[-1] for i in glob.glob(os.path.join(self.datadir, f"two_view_{self.split_folder}") + "/*") if os.path.isdir(i)]

        self.img_left_filenames   = [os.path.join(self.datadir, f"two_view_{self.split_folder}", img, "im0.png")     for img in img_list]  
        self.img_right_filenames  = [os.path.join(self.datadir, f"two_view_{self.split_folder}", img, "im1.png")     for img in img_list]
        self.disp_left_filenames  = [os.path.join(self.datadir, f"two_view_{self.split_folder}", img, "disp0GT.pfm") for img in img_list]
        # self.disp_right_filenames = [os.path.join(self.datadir, f"two_view_{self.split_folder}", img, "disp1GT.pfm") for img in img_list]

    def load_image(self, filename) -> Image:
        return Image.open(filename).convert('RGB')

    def load_disp(self, path) -> np.ndarray[np.float32]:
        if '.png' in path:
            data = Image.open(path)
            data = np.ascontiguousarray(data,dtype=np.float32)/256
        else:
            data = cfnet.readpfm.readPFM(path)[0]
            data = np.ascontiguousarray(data, dtype=np.float32)
        
        data[data == np.inf] = 0
        return data
        

    def __getitem__(self, index):

        img_left = self.load_image(self.img_left_filenames[index])
        img_right = self.load_image(self.img_right_filenames[index])
        
        # Disparity doen't exist for test set
        disp_left = self.load_disp(self.disp_left_filenames[index]) if self.training else None
        # disp_right = self.load_disp(self.disp_right_filenames[index])
        # occ_left = self.load_occ(self.occ_left_filenames[index])
        # occ_right = self.load_occ(self.occ_right_filenames[index])
        
        if self.model_name == 'cfnet':
            return self.get_item_cfnet(img_left, img_right, disp_left, index)
            
        elif self.model_name == 'psmnet':
           raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        elif self.model_name == 'gwcnet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        elif self.model_name == 'sttr':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        elif self.model_name == 'hsmnet':
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")
        
        else:
            raise NotImplemented(f"No dataloder for {self.model_name} implemented")

        
        

    def get_item_cfnet(self, left_img, right_img, disparity, index:int):

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
            if h % 64 == 0:
                top_pad = 0
            else:
                top_pad = 64 - (h % 64)

            if w % 64 == 0:
                right_pad = 0
            else:
                right_pad = 64 - (w % 64)
            assert top_pad >= 0 and right_pad >= 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            print(f"top_pad: {top_pad}, right_pad: {right_pad}")
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
                        "right_pad": right_pad,
                        "left_filename": self.img_left_filenames[index],
                        "right_filename": self.img_right_filenames[index]}

    def __len__(self):
        return len(self.img_left_filenames)
