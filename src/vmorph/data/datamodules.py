import os
from typing import List, Dict
import model.lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
import nibabel as nib
import random
import json
import matplotlib.pyplot as plt
from data.datasets import ContrastiveImagingAndECGDataset

class CardiacUkbbDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Dict):
        #  fold: str = "./", img_size: tuple = (32, 32, 32), img_spacing: tuple = (1, 1, 1)):
        #  dataset='images', use_mask=True, coordinate_batch_size=10000):
        super().__init__()
        self.batch_size = hparams.data.batch_size
        self.num_workers = hparams.data.num_workers
        self.train_path = hparams.data.train_path
        self.train_seg_path = hparams.data.train_seg_path
        self.val_seg_path = hparams.data.val_seg_path
        self.val_path = hparams.data.val_path
        self.del_seg = hparams.data.del_seg
        self.img_size = hparams.data.img_size


    def setup(self, stage: List[str] = None):
        # This method expects a stage argument. It is used to separate setup logic for
        # trainer.{fit,validate,test,predict}. If setup is called with stage=None,
        # we assume all stages have been set-up.

        # setup data transformations
        # self.setup_transform()

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            self.train_dataset = ContrastiveImagingAndECGDataset(self.train_path,
                                                                 self.train_seg_path,
                                                                 delete_segmentation=self.del_seg,
                                                                 img_size=self.img_size)
            self.val_dataset = ContrastiveImagingAndECGDataset(self.val_path,
                                                               self.val_seg_path,
                                                               delete_segmentation=self.del_seg,
                                                               img_size=self.img_size)
        # Assign Test split(s) for use in Dataloaders
        # if stage in (None, "test"):

            # self.nlst_test = NLSTDataset(f'{self.data_dir}/NLST', self.config['testing'], transforms=None,
            #                              augm_transforms=self.val_augm_transforms)

        # if stage in (None, "predict"):
        #     self.nlst_predict = None
        #     print()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.nlst_test, batch_size=self.batch_size, num_workers=32)

    # def predict_dataloader(self):
    #     return DataLoader(self.nlst_predict, batch_size=self.batch_size, num_workers=32)

    # def setup_transform(self):
    #     transforms = [
    #         tio.Resample(self.img_spacing, image_interpolation='linear'),
    #         tio.CropOrPad(target_shape=self.img_size),
    #         tio.ZNormalization(masking_method=tio.ZNormalization.mean, exclude=('FixedMask', 'MovingMask')),
    #         # whitening
    #
    #     ]
    #     augm_transforms = [
    #         tio.RandomAffine(image_interpolation='linear'),
    #
    #     ]
    #     self.transforms = tio.Compose(transforms)
    #     self.augm_transforms = tio.Compose(augm_transforms)

    # def __len__(self):
    # return len(self.filenames)

