import numpy as np
import omegaconf
import os
import pandas as pd
import SimpleITK as sitk
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from skimage.measure import block_reduce
from transformers import CLIPTokenizer
from tqdm import tqdm

def setup_dataset(train_ids: pd.DataFrame, val_ids: pd.DataFrame, batch_size: int, context: str = None, image_size: tuple = (192, 256, 192), downsample: bool = True, labels: bool=False) -> tuple:
    """
    Prepare dataloaders for training.

    Args:
        train_ids (pd.DataFrame): The file location of the spreadsheet
        val_ids (pd.DataFrame): A flag used to print the columns to the console (default is False)
        batch_size (int): The size of batches to be used for training and validation
        context (bool, optional): A flag used to speficy whether to use context/conditioning or not (default is False)

    Returns:
        The training and validation dataloaders.
    """

    ukbb_dm = UKBBDataModule(train_csv=train_ids, val_csv=val_ids, batch_size=batch_size, context=context, image_size=image_size, downsample=downsample , labels=labels)
    ukbb_dm.setup(stage="fit")

    train_dl = ukbb_dm.train_dataloader()
    val_dl = ukbb_dm.val_dataloader()
    return train_dl, val_dl

def get_testset(test_ids: pd.DataFrame, batch_size: int, context=None, image_size: tuple = (192, 256, 192), downsample: bool = True):
    ukbb_dm = UKBBDataModule(test_csv=test_ids, batch_size=batch_size, context=context, image_size=image_size, downsample=downsample)
    ukbb_dm.setup(stage="inference")

    test_dl = ukbb_dm.test_dataloader()
    return test_dl

class UKBBDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv: pd.DataFrame = None,
                 val_csv: pd.DataFrame = None,
                 data_dir: str=None,
                 test_csv: pd.DataFrame = None,
                 phase: str='train',
                 context: str=None,
                 ispath: bool=True,
                 image_size: tuple=(192, 256, 192),
                 downsample: bool=False,
                 labels: bool=False,
                 batch_size: int=4):
        super().__init__()
        self.data_dir = data_dir

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.phase = phase
        self.batch_size = batch_size
        self.context=context
        self.ispath = ispath
        self.image_size = image_size
        self.downsample = downsample
        self.DatasetClass = UKBB
        self.labels = labels


    def setup(self, stage: str = None) -> None:
        # assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.phase == 'train':

                self.ukbb_train = self.DatasetClass(self.data_dir, self.train_csv, self.context, self.image_size, self.downsample, ispath=self.ispath, labels = self.labels)
                self.ukbb_val = UKBB(self.data_dir, self.val_csv, self.context, self.image_size, self.downsample, ispath=self.ispath, labels = self.labels)
        if stage == "inference":
            if self.phase == "test":
                self.ukbb_test = UKBB(self.data_dir, self.test_csv, self.context, self.image_size, self.downsample, ispath=self.ispath, labels = self.labels)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.ukbb_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ukbb_val, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ukbb_test, batch_size=self.batch_size, shuffle=True, num_workers=8)


class UKBB(Dataset):
    def __init__(self, data_dir, csv_dir, context, image_size = (192, 256, 192), downsample=True, ispath=True, labels=False):
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.context = context
        self.labels = labels
        self.downsample = downsample
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_size = image_size
        if ispath:
            self.data_df = pd.read_csv(csv_dir)
            self.full_df = pd.read_csv(csv_dir)
        else:
            self.data_df = csv_dir
            self.full_df = csv_dir

        # if conditioning flag is set, load the condition list and tokenizer
        if self.context:
            # self.condition_list = self.data_df[["csf_volume"]].values.tolist()
            self.condition_list = self.data_df[[self.context]].values.tolist()
            self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")


    def __len__(self):
        return len(self.data_df)

    def __get_neighbourhood__(self, context, batch_size, neighborhood_size=25):
        neighborhood = self.full_df[np.isclose(self.full_df[self.context], context, atol=0.05)].copy()
        # print(neighborhood)
        if len(neighborhood) < neighborhood_size:
            # if neigh too small, we increase the tolerance
            neighborhood = self.full_df[np.isclose(self.full_df[self.context], context, atol=0.10)].copy()
        if len(neighborhood) < neighborhood_size:
            # if neigh still too small, we reduce the neighborhood size
            neighborhood = neighborhood.sample(n=len(neighborhood)).reset_index()
        else:
            neighborhood = neighborhood.sample(n=neighborhood_size).reset_index()

        ds = UKBB(self.data_dir, neighborhood, self.context, image_size=self.image_size, downsample=self.downsample, ispath=False, labels=self.labels)

        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)


    def __getitem__(self, idx):
        # read image and labels
        image_name = self.data_df.loc[idx, 'image_path'].replace('affine', 'rigid')

        # check if image file exists
        assert os.path.isfile(image_name)

        #load image
        im = load_nifty(image_name, image_size=self.image_size)

        if self.labels:
            label_path = self.data_df.loc[idx, 'label_path'].replace('affine', 'rigid')
            label = load_nifty(label_path, image_size=self.image_size)

        if self.downsample:
            im = torch.Tensor(block_reduce(im, block_size=2))

        im = torch.clip(im, min=0, max=2000)
        image_tensor_normalized = normalise_intensity(im, mode="minmax", min_in=0, max_in=2000)

        if self.context:
            # get the condition matching the current image
            context = self.condition_list[idx]
            if self.context == "csf_volume":
                prompt = f"T1-weighted image of a brain with a ventricle size of {context[0]:.3f}."
            else:
                prompt = f"T1-weighted image of a {int(100*context[0])} years old brain."
            # tokenize the condition
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            if self.labels:
                return image_tensor_normalized, text_inputs.input_ids, label
            return image_tensor_normalized, text_inputs.input_ids, categorise(context[0])
        if self.labels:
            return image_tensor_normalized, label
        return image_tensor_normalized

def categorise(x):
    if x<0.1:
        return 0.1
    elif x>0.9:
        return 0.9
    else:
        return round(x, 1)

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
    """
    Loads torch.tensor image from disk and crops it to image_size.

    Args:
        image_path (str): path to the image.
        image_size (int, int, int): desired size of the image.

    Returns:
        sitk.Image
    """

    image = sitk.ReadImage(image_path)
    image = sitk.DICOMOrient(image, 'RPI')
    image = sitk.GetArrayFromImage(image)
    if image_size:
        image = crop_and_pad(image.reshape(1, *image.shape), new_size=image_size)

    image = torch.tensor(image)

    return image

def load_sitk(path):
    """
    Loads sitk.Image from disk.

    Args:
        path (str): path to the image.

    Returns:
        sitk.Image
    """
    im = sitk.DICOMOrient(sitk.ReadImage(path), 'RIP')
    return im


def np2sitk(np_image, sitk_img):
    """
    This function takes a numpy array and casts it to a SimpleItk image.

    Args:
        np_image (np.array): numpy image to convert.
        sitk_img (sitk.Image): sitk Image containing params.

    Returns:
        sitk.Image: Corresponding sitk.Image
    """
    new_sitk_img = sitk.GetImageFromArray(np_image)
    new_sitk_img.SetOrigin(sitk_img.GetOrigin())
    new_sitk_img.SetSpacing(sitk_img.GetSpacing())
    new_sitk_img.SetDirection(sitk_img.GetDirection())
    return new_sitk_img

def save_sitk(outpath, arr, sitk_arr):
    """
    Save image to disk.

    Args:
        outpath (str): path to file.
        arr (np.array): image array to save.
        sitk_arr (sitk.Image): sitk Image object corresponding.
    """
    sitk_im = np2sitk(arr, sitk_arr)
    sitk.WriteImage(sitk_im, outpath)