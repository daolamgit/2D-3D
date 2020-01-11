import os.path
import glob
from os.path import join
import h5py

from skimage.transform import resize

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
        #pt_index to load the patient

        self.opt = opt
        self.root = opt.dataroot

        self.images = []
        self.masks = []


        pt_list     = sorted(glob.glob( join(opt.dataroot, '*.hdf5')))
        pt_path     = pt_list[opt.pt_index]
        f_h5        = h5py.File( pt_path, 'r')
        self.images      = np.asarray(f_h5['Volume_resize'], dtype=np.float32)
        masks       = np.asarray(f_h5['Masks_resize'], dtype=np.int)

        #convert masks from S,W,H to S, Class, W, H

        C = np.amax(masks)
        self.masks = np.eye(C+1)[masks] #very subtle
        #roll axis
        self.masks = np.moveaxis(self.masks, -1, 1)


        #resize to 256, remember to take it out
        Shape = self.images.shape
        self.images = resize( self.images, (Shape[0], 256, 256))
        self.masks = resize(self.masks, (Shape[0], 256, 256))

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase, 'Images')
        # # go through directory return os.path for all images
        # slice_filetype = ['.npy']
        # self.AB_paths = sorted(make_dataset(self.dir_AB, slice_filetype))
        #
        # # assert self.opt.loadSize == self.opt.fineSize, 'No resize or cropping.'



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
        input_nc = self.opt.input_nc #now 1
        output_nc = self.opt.output_nc #now 6

        # AB_path = self.AB_paths[index]
        # mat = loadmat(AB_path)

        # image   = np.load( AB_path).astype(np.float32)
        image = np.expand_dims( self.images[index], axis=0)
        masks = self.masks[index]
        # BA_path = AB_path.replace("Images", "Dose")
        # dose    = np.expand_dims(np.load( BA_path).astype(np.float32), axis=0)

        #some interp resize
        #Augmentation
        # if self.needAugment:
        #
        #
        #
        #     #convert image to channel last as opencv
        #     image_l = np.transpose( image, (1,2,0)).astype( bool)
        #     dose_l = dose.squeeze()
        #
        #     # import matplotlib.pyplot as plt
        #     # plt.imshow(dose_l, cmap='Accent')
        #     # plt.show()
        #
        #     segmap = SegmentationMapOnImage(image_l, nb_classes=6, shape=image.shape)
        #
        #     dose_aug, image_aug = self.seq( image= dose_l, segmentation_maps=segmap)
        #
        #     dose = np.expand_dims(dose_aug, axis=0)
        #     image = image_aug.arr.transpose((2,0,1)) #put it back


        # return {'A': image, 'B': dose, 'A_paths': AB_path, 'B_paths': BA_path}
        return {'A': image, 'B': masks, 'A_paths': [], 'B_paths': []}


    def __len__(self):
        return len(self.images)

    def name(self):
        return 'MySliceDataset'
