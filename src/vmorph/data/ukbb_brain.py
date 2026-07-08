import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import os
import pandas as pd
from torchvision import transforms
import logging
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import omegaconf
import os
import pandas as pd
from time import time
import skimage.io as io
from skimage.measure import block_reduce

def setup_dataset(batch_size):
    ukbb_dm = UKBBDataModule(batch_size=batch_size)
    ukbb_dm.setup(stage="fit")

    train_dl = ukbb_dm.train_dataloader()
    val_dl = ukbb_dm.val_dataloader()
    return train_dl, val_dl

class UKBBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str,
                 train_csv: str,
                 val_csv: str,
                 test_csv: str,
                 phase: str='train',
                 batch_size: int=4):
        super().__init__()
        self.data_dir = data_dir

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        # self.sets = parse_settings()
        self.phase = phase
        self.batch_size = batch_size


    def setup(self, stage: str = None) -> None:
        # assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.phase == 'train':
                self.train_dataset = UKBB(self.data_dir, self.train_csv)
                self.val_dataset = UKBB(self.data_dir, self.val_csv)
        if stage == "inference":
            if self.phase == "test":
                self.ukbb_test = UKBB(self.data_dir, self.test_csv)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.ukbb_test, batch_size=self.batch_size, shuffle=True, num_workers=8)


class UKBB(Dataset):
    def __init__(self, data_dir, csv_dir):
        self.data_dir = data_dir
        self.csv_dir = csv_dir

        self.transform = transforms.Compose([transforms.ToTensor()])

        fixed_imaging_df = pd.read_csv(csv_dir)

        self.fixed_imaging_df = fixed_imaging_df
        self.moving_imaging_df = self.fixed_imaging_df.copy().sample(frac=1).reset_index()
        self.fixed_labels_list = self.fixed_imaging_df[["age"]].values.tolist()
        self.moving_labels_list = self.moving_imaging_df[["age"]].values.tolist()


    def __len__(self):
        return len(self.fixed_imaging_df)

    def __getitem__(self, idx):
        data_dict = {}
        # read image and labels
        fixed_image_path = self.fixed_imaging_df.loc[idx, 'image_path']
        fixed_seg_path = self.fixed_imaging_df.loc[idx, 'seg_path']
        fixed_mask_path = self.fixed_imaging_df.loc[idx, 'mask_path']
        moving_image_path = self.moving_imaging_df.loc[idx, 'image_path']
        moving_seg_path = self.moving_imaging_df.loc[idx, 'seg_path']
        moving_mask_path = self.moving_imaging_df.loc[idx, 'mask_path']
        
        # fixed_label = torch.tensor(self.fixed_labels_list[idx], dtype=torch.float32)
        # moving_label = torch.tensor(self.moving_labels_list[idx], dtype=torch.float32)

        # check if image file exists
        assert os.path.isfile(fixed_image_path), f"The file '{fixed_image_path}' does not exist."
        assert os.path.isfile(moving_image_path), f"The file '{moving_image_path}' does not exist."

        #load image
        fixed_image = self.load_image(fixed_image_path)
        fixed_seg = self.load_image(fixed_seg_path)
        fixed_mask = self.load_image(fixed_mask_path)
        moving_image = self.load_image(moving_image_path)
        moving_seg = self.load_image(moving_seg_path)
        moving_mask = self.load_image(moving_mask_path)

        data_dict['fixed'] = fixed_image
        data_dict['moving'] = moving_image
        data_dict['fixed_seg'] = fixed_seg
        data_dict['moving_seg'] = moving_seg
        data_dict['fixed_mask'] = fixed_mask
        data_dict['moving_mask'] = moving_mask
        # data_dict['fixed_original'] = fixed_image
        # data_dict['moving_original'] = moving_image

        return data_dict


    def load_image(self, image_name, img_size=(160, 224, 160), clip=True, dowsampling=False, normalise=True, max_intensity=2000):
        # load and crop/pad to the desired dimension
        im = load_nifty(image_name, image_size=img_size)
        
        if clip:
            im = torch.clip(im, min=0, max=max_intensity)
        # downsampling
        if dowsampling:
            im = block_reduce(np.array(im), block_size=2)
        if normalise:
            im = normalise_intensity(torch.Tensor(im), mode="minmax", min_in=0, max_in=max_intensity)
        return im 


    def convert_nii_to_tensor(self, image):
        [z,y,x] = image.shape
        image_tensor = np.reshape(image, [1,z,y,x])
        image_tensor = image_tensor.astype("float32")
        return image_tensor


def normalise_intensity(x,
                        mode="minmax",
                        min_in=0.0,
                        max_in=255.0,
                        min_out=0.0,
                        max_out=1.0,
                        clip=False,
                        clip_range_percentile=(0.05, 99.95),
                        ):
    """
    Intensity normalisation (& optional percentile clipping)
    for both Numpy Array and Pytorch Tensor of arbitrary dimensions.

    The "mode" of normalisation indicates different ways to normalise the intensities, including:
    1) "meanstd": normalise to 0 mean 1 std;
    2) "minmax": normalise to specified (min, max) range;
    3) "fixed": normalise with a fixed ratio

    Args:
        x: (ndarray / Tensor, shape (N, *size))
        mode: (str) indicate normalisation mode
        min_in: (float) minimum value of the input (assumed value for fixed mode)
        max_in: (float) maximum value of the input (assumed value for fixed mode)
        min_out: (float) minimum value of the output
        max_out: (float) maximum value of the output
        clip: (boolean) value clipping if True
        clip_range_percentile: (tuple of floats) percentiles (min, max) to determine the thresholds for clipping

    Returns:
        x: (same as input) in-place op on input x
    """

    # determine data dimension
    dim = x.ndim - 1
    image_axes = tuple(range(1, 1 + dim))  # (1,2) for 2D; (1,2,3) for 3D

    # for numpy.ndarray
    if type(x) is np.ndarray:
        # Clipping
        if clip:
            # intensity clipping
            clip_min, clip_max = np.percentile(x, clip_range_percentile, axis=image_axes, keepdims=True)
            x = np.clip(x, clip_min, clip_max)

        # Normalise meanstd
        if mode == "meanstd":
            mean = np.mean(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            std = np.std(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            min_in = np.amin(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            max_in = np.amax(x, axis=image_axes, keepdims=True)  # (N, *range(dim)))
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12) + min_out # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not understood."
                             "Expect either one of: 'meanstd', 'minmax', 'fixed'")

        # cast to float 32
        x = x.astype(np.float32)

    # for torch.Tensor
    elif type(x) is torch.Tensor:
        # todo: clipping not supported at the moment (requires Pytorch version of the np.percentile()

        # Normalise meanstd
        if mode == "meanstd":
            mean = torch.mean(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            std = torch.std(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            # get min/max across dims by flattening first
            min_in = x.flatten(start_dim=1, end_dim=-1).min(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            max_in = x.flatten(start_dim=1, end_dim=-1).max(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12) + min_out  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not recognised."
                             "Expect: 'meanstd', 'minmax', 'fixed'")

        # cast to float32
        x = x.float()

    else:
        raise TypeError("Input data type not recognised, support numpy.ndarray or torch.Tensor")
    return x

def crop_and_pad(x, new_size=192, mode="constant", **kwargs):
    """
    Crop and/or pad input to new size.
    (Adapted from DLTK: https://github.com/DLTK/DLTK/blob/master/dltk/io/preprocessing.py)

    Args:
        x: (np.ndarray) input array, shape (N, H, W) or (N, H, W, D)
        new_size: (int or tuple/list) new size excluding the batch size
        mode: (string) padding value filling mode for numpy.pad() (compulsory in Numpy v1.18)
        kwargs: additional arguments to be passed to np.pad

    Returns:
        (np.ndarray) cropped and/or padded input array
    """
    assert isinstance(x, (np.ndarray, np.generic))
    new_size = param_ndim_setup(new_size, ndim=x.ndim - 1)

    dim = x.ndim - 1
    sizes = x.shape[1:]

    # Initialise padding and slicers
    to_padding = [[0, 0] for i in range(x.ndim)]
    slicer = [slice(0, x.shape[i]) for i in range(x.ndim)]

    # For each dimensions except the dim 0, set crop slicers or paddings
    for i in range(dim):
        if sizes[i] < new_size[i]:
            to_padding[i+1][0] = (new_size[i] - sizes[i]) // 2
            to_padding[i+1][1] = new_size[i] - sizes[i] - to_padding[i+1][0]
        else:
            # Create slicer object to crop each dimension
            crop_start = int(np.floor((sizes[i] - new_size[i]) / 2.))
            crop_end = crop_start + new_size[i]
            slicer[i+1] = slice(crop_start, crop_end)

    return np.pad(x[tuple(slicer)], to_padding, mode=mode, **kwargs)

def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param

def load_nifty(image_path, image_size=None):

    image = sitk.ReadImage(image_path)
    image = sitk.DICOMOrient(image, 'RPI')
    image = sitk.GetArrayFromImage(image)
    if image_size:
        image = crop_and_pad(image.reshape(1, *image.shape), new_size=image_size)

    image = torch.tensor(image)

    return image