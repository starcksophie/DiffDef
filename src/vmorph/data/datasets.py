import os
import random

from torch.utils.data import Dataset
from vmorph.data.utils_d import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor, _load2d, _magic_slicer
import matplotlib.pyplot as plt

class _BaseDataset(Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))
        self.data_path_dict = dict()

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 evaluate=False,
                 modality='t1t1',
                 atlas_path=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['fixed', 'moving']

        self.modality = modality
        self.atlas_path = atlas_path

    # def _set_path(self, index):
    #     # choose the fixed and moving subjects/paths
    #     if self.atlas_path is None:
    #         self.tar_subj_id = self.subject_list[index]
    #         self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
    #     else:
    #         self.tar_subj_path = self.atlas_path
    #
    #     self.src_subj_id = random.choice(self.subject_list)
    #     self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'
    #
    #     self.data_path_dict['fixed'] = f'{self.tar_subj_path}/T1_brain.nii.gz'
    #
    #     # modality
    #     if self.modality == 't1t1':
    #         self.data_path_dict['moving'] = f'{self.src_subj_path}/T1_brain.nii.gz'
    #     elif self.modality == 't1t2':
    #         self.data_path_dict['moving'] = f'{self.src_subj_path}/T2_brain.nii.gz'
    #     else:
    #         raise ValueError(f'Modality ({self.modality}) not recognised.')
    #
    #     # eval data
    #     if self.evaluate:
    #         # T1w image of moving subject for visualisation
    #         self.data_path_dict['fixed_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'
    #         self.img_keys.append('fixed_original')
    #         # segmentation
    #         self.data_path_dict['fixed_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
    #         self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
    def _set_path(self, index):
        # choose the fixed and moving subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict['fixed'] = f'{self.tar_subj_path}/T1_brain_norm.nii.gz'

        # modality
        if self.modality == 't1t1':
            self.data_path_dict['moving'] = f'{self.src_subj_path}/T1_brain_norm.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['moving'] = f'{self.src_subj_path}/T2_brain_norm.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        if self.modality == 't1t1':
            self.data_path_dict['moving_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['fixed_mask'] = f'{self.src_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['fixed_original'] = f'{self.tar_subj_path}/T1_brain_norm.nii.gz'
            self.data_path_dict['moving_original'] = f'{self.src_subj_path}/T1_brain_norm.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['fixed_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['moving_mask'] = f'{self.src_subj_path}/T2_brain_mask.nii.gz'
            self.data_path_dict['fixed_original'] = f'{self.tar_subj_path}/T2_brain_norm.nii.gz'
            self.data_path_dict['moving_original'] = f'{self.src_subj_path}/T1_brain_norm.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        self.img_keys.append('fixed_original')
        self.img_keys.append('moving_original')
        # eval data
        # if self.evaluate:
            # T1w image of moving subject for visualisation
            # self.data_path_dict['fixed_original'] = f'{self.tar_subj_path}/T2_brain.nii.gz'
            # self.data_path_dict['moving_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            # segmentation
        self.data_path_dict['fixed_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
        self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        return _to_tensor(data_dict)


class CardiacMR2D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 evaluate=False,
                 slice_range=None,
                 slicing=None,
                 crop_size=(192, 192),
                 batch_size=None,
                 ):
        super(CardiacMR2D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['fixed', 'moving']

        self.slice_range = slice_range
        self.slicing = slicing
        if batch_size is not None:
            self.subject_list = self.subject_list * batch_size

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['fixed'] = f'{self.subj_path}/sa_ED.nii.gz'
        self.data_path_dict['moving'] = f'{self.subj_path}/sa_ES.nii.gz'
        if self.evaluate:
            self.data_path_dict['fixed_original'] = self.data_path_dict['moving']
            self.img_keys.append('fixed_original')
            self.data_path_dict['fixed_seg'] = f'{self.subj_path}/label_sa_ED.nii.gz'
            self.data_path_dict['moving_seg'] = f'{self.subj_path}/label_sa_ES.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load2d(self.data_path_dict)
        data_dict = _magic_slicer(data_dict, slice_range=self.slice_range, slicing=self.slicing)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        return _to_tensor(data_dict)


from typing import List, Tuple, Dict
import random
import csv

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
from torchvision.io import read_image
# import utils.ecg_augmentations as augmentations


class ContrastiveImagingAndECGDataset(Dataset):
    """
    Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.
    The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
    The first ECG view is never augmented. The second view is corrupted by replacing {corruption_rate} features
    with values chosen from the empirical marginal distribution of that feature.
    """

    def __init__(
            self,
            data_path_imaging: str, data_path_seg: str, delete_segmentation: bool, img_size: int = 210, augmentation_rate: float = 0) -> None:
            # delete_segmentation: bool, augmentation: transforms.Compose,
            # augmentation_rate: float,
            # data_path_ecg: str, ecg_random_crop: bool,
            # labels_path: str, img_size: int,
            # args) -> None:
        # Imaging
        self.data_imaging = torch.load(data_path_imaging)
        self.data_segmentations = torch.load(data_path_seg)
        # self.data_seg = torch.load(data_path_imaging)
        # self.transform = augmentation
        self.delete_segmentation = delete_segmentation
        self.augmentation_rate = augmentation_rate

        if self.delete_segmentation:
            self.data_imaging = [image[0::2, ...] for image in self.data_imaging]
        self.img_size = img_size

        self.default_transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size), antialias=None),
            transforms.Lambda(lambda x: x.float())
        ])
        # ECG
        # self.data_ecg = torch.load(data_path_ecg)
        # self.data_ecg = [d.unsqueeze(0) for d in self.data_ecg]
        # self.data_ecg = [d[:, :args.input_electrodes, :] for d in self.data_ecg]
        # self.ecg_random_crop = ecg_random_crop
        # Classifier
        # self.labels = torch.load(labels_path)
        # self.args = args


    def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
        """
        Generates two views of a subjects image. Also returns original image resized to required dimensions.
        The first is always augmented. The second has {augmentation_rate} chance to be augmented.
        """
        im = self.data_imaging[index]
        seg = self.data_segmentations[index]
        # im = transforms.CenterCrop(size=int(0.75*self.img_size))(im)
        im = torchvision.transforms.functional.crop(im, top=int(0.21 * self.img_size), left=int(0.325 * self.img_size),
                                                    height=int(0.375 * self.img_size), width=int(0.375 * self.img_size))

        seg = torchvision.transforms.functional.crop(seg, top=int(0.21 * self.img_size), left=int(0.325 * self.img_size),
                                                    height=int(0.375 * self.img_size), width=int(0.375 * self.img_size))
        # if random.random() < self.augmentation_rate:
        #     im_aug = (self.transform(im))
        # else:
        #     im_aug = (self.default_transform(im))
        #
        im = self.default_transform(im)
        seg = self.default_transform(seg)

        return [im, seg]

    # def generate_ecg_views(self, index: int) -> List[torch.Tensor]:
    #     """
    #     Generates two views of a subjects ECG. The first is always the augmented.
    #     """
    #     data = self.data_ecg[index]
    #
    #     if self.ecg_random_crop:
    #         transform = augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False)
    #     else:
    #         transform = augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False)
    #     ecg_orig = transform(data)
    #
    #     ecg_aug = ecg_orig
    #     if random.random() < self.augmentation_rate:
    #         augment = transforms.Compose([augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise),
    #                                       augmentations.Jitter(sigma=self.args.jitter_sigma),
    #                                       augmentations.Rescaling(sigma=self.args.rescaling_sigma),
    #                                       # augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
    #                                       # augmentations.TimeFlip(prob=0.33),
    #                                       # augmentations.SignFlip(prob=0.33)
    #                                       ])
    #         ecg_aug = augment(ecg_aug)
    #
    #     return ecg_aug, ecg_orig


    def __getitem__(self, index: int) -> Dict:
        im, seg = self.generate_imaging_views(index)
        # ecg_aug, ecg_orig = self.generate_ecg_views(index)
        # label = torch.tensor(self.labels[index], dtype=torch.long)
        # return image_aug, ecg_aug, label, image_orig, ecg_orig, index
        # return image_aug[0::2, ...], ecg_aug, label, image_orig[0::2, ...], ecg_orig, index
        # return image_aug[0::2, ...], image_orig[0::2, ...]
        data_dict = {}
        data_dict['fixed'] = im[0].unsqueeze(0)
        data_dict['moving'] = im[1].unsqueeze(0)
        data_dict['fixed_original'] = data_dict['moving']
        data_dict['fixed_seg'] = seg[0].unsqueeze(0)
        data_dict['moving_seg'] = seg[1].unsqueeze(0)

        return data_dict

    def __len__(self) -> int:
        return len(self.data_imaging)

