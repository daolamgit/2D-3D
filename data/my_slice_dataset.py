import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from scipy.io import loadmat
from skimage.transform import rescale
import pudb
import numpy as np

import Augmentor

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

class MySliceDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase, 'Images')
        # go through directory return os.path for all images
        slice_filetype = ['.npy']
        self.AB_paths = sorted(make_dataset(self.dir_AB, slice_filetype))
        # assert self.opt.loadSize == self.opt.fineSize, 'No resize or cropping.'

        # self.augment = Aug
        self.needAugment = 0
        if self.opt.phase == 'train':
            if self.opt.Augment:
                self.needAugment = 1
                self.seq = iaa.Sequential( [
                iaa.Affine( rotate= (-3, 3),
                            scale=(.95, 1.05)),
            ])
            else:
                self.needAugment = 0
        else:
            self.needAugment = 0

    def __getitem__(self, index):
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        AB_path = self.AB_paths[index]
        # mat = loadmat(AB_path)

        image   = np.load( AB_path).astype(np.float32)
        BA_path = AB_path.replace("Images", "Dose")
        dose    = np.expand_dims(np.load( BA_path).astype(np.float32), axis=0)

        #some interp resize
        #Augmentation
        if self.needAugment:



            #convert image to channel last as opencv
            image_l = np.transpose( image, (1,2,0)).astype( bool)
            dose_l = dose.squeeze()

            # import matplotlib.pyplot as plt
            # plt.imshow(dose_l, cmap='Accent')
            # plt.show()

            segmap = SegmentationMapOnImage(image_l, nb_classes=6, shape=image.shape)

            dose_aug, image_aug = self.seq( image= dose_l, segmentation_maps=segmap)

            dose = np.expand_dims(dose_aug, axis=0)
            image = image_aug.arr.transpose((2,0,1)) #put it back


        return {'A': image, 'B': dose, 'A_paths': AB_path, 'B_paths': BA_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'SliceDataset'
