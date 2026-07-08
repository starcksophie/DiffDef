import torch
import numpy as np
import os
from data.ukbb_brain import normalise_intensity, crop_and_pad
import SimpleITK as sitk
from deepali.core import Grid
from deepali.losses import functional as L
import deepali.spatial as spatial
from tqdm import tqdm
import argparse
import os
from typing import Any, Callable, Optional, Tuple, Type, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import torch
from torch import Tensor, optim
from deepali.core.environ import cuda_visible_devices
from skimage.measure import block_reduce


def main(data_dir, verbose=False):
    # Use first device specified in CUDA_VISIBLE_DEVICES if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda_visible_devices() else "cpu")
    print(device)

    subjects = os.listdir(data_dir)
    T1_NMI_path = f"{data_dir}/{subjects[0]}/T1/T1_brain_to_MNI.nii.gz"
    T1_brain_path = f"{data_dir}/{subjects[0]}/T1/T1_unbiased_brain.nii.gz"

    for i  in tqdm(range(len(subjects))):

        try:
            if os.path.exists(f"{data_dir}/{subjects[i]}/T1/unusable"):
                # print(f"{data_dir}/{subjects[i]}")
                continue

            affine_dir = f"{data_dir}/{subjects[i]}/T1/rigid_reg"

            if not os.path.exists(affine_dir):
                os.makedirs(affine_dir)

            if len(affine_dir) == 0:
                continue

            T1_brain_mask_path = f"{data_dir}/{subjects[i]}/T1/T1_brain_mask.nii.gz"
            T1_brain_seg_path = f"{data_dir}/{subjects[i]}/T1/T1_fast/T1_brain_seg.nii.gz"

            fixed_path = f"{data_dir}/{subjects[0]}/T1/T1_brain_to_MNI.nii.gz"
            moving_path = f"{data_dir}/{subjects[i]}/T1/T1_unbiased_brain.nii.gz"

            # load the images using simple itk and a fixed orientation
            fixed, image_sitk = load_image(fixed_path, reorient=True, clip=True, downsample=False, norm_int=True)
            moving, _ = load_image(moving_path, reorient=True, clip=True, downsample=False, norm_int=True)
            moving_seg, _ = load_image(T1_brain_seg_path, reorient=True, clip=False, downsample=False, norm_int=False)
            moving_mask, _ = load_image(T1_brain_mask_path, reorient=True, clip=False, downsample=False, norm_int=False)

            # copy data to device
            fixed = fixed.to(device)
            moving = moving.to(device)
            moving_seg = moving_seg.to(device)
            moving_mask = moving_mask.to(device)

            # define a grid for deepali
            grid = Grid(shape=fixed.shape[1:], device=device)

            if verbose:
                print("size:     ", list(grid.size()))
                print("origin:   ", grid.origin().tolist())
                print("center:   ", grid.center().tolist())
                print("spacing:  ", grid.spacing().tolist())
                print("direction:", grid.direction().tolist())

            # define a loss function
            sim_loss = L.mse_loss
            # sim_loss = L.ncc_loss

            # define an affine transformation - not multi-resolution for now
            affine = spatial.RigidTransform(grid)

            # have 2 different grid samplers: one with linear interp and one with nearest
            transformer = spatial.ImageTransformer(affine).to(device=device)
            transformer_seg = spatial.ImageTransformer(affine, sampling='nearest').to(device=device)

            # optimizer and iterations
            optimizer = optim.Adam(transformer.parameters(), lr=1e-2)
            iterations = 150

            # create a list for the losses in case we want to plot the registration loss to assess the performance
            loss_list = []
            for _ in range(iterations):
                warped_batch = transformer(moving)
                loss = sim_loss(warped_batch, fixed)
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # warp using the transformers
            with torch.inference_mode():
                warped: Tensor = transformer(moving)[0]
                warped_seg: Tensor = transformer_seg(moving_seg)[0]
                warped_mask: Tensor = transformer_seg(moving_mask)[0]

            # define the paths to save the warped images and save them using simple itk
            T1_path = f"{affine_dir}/T1_un_norm.nii.gz"
            T1_seg_path = f"{affine_dir}/T1_seg.nii.gz"
            T1_mask_path = f"{affine_dir}/T1_mask.nii.gz"

            save_sitk(T1_path, warped.cpu().numpy(), image_sitk)
            # save_sitk(T1_path_NMI, fixed[0]. is successfully created. \ncpu().numpy(), image_sitk)
            save_sitk(T1_seg_path, warped_seg.cpu().numpy(), image_sitk)
            save_sitk(T1_mask_path, warped_mask.cpu().numpy(), image_sitk)

        except:
            continue


    print('\nSuccess!\n')

def load_image(image_path, img_size=(160, 224, 160), reorient=False, clip=False, norm_int=False, downsample=False):
    # load and crop/pad to the desired dimension
    im, im_sitk = load_nifty(image_path, image_size=img_size, reorient=reorient)
    # TODO: define a sutable upper boundary for this
    if clip:
        im = torch.clip(im, min=0, max=2000)
    # downsampling
    if downsample:
        im = block_reduce(np.array(im), block_size=2)
    if norm_int:
        im = normalise_intensity(torch.Tensor(im), mode="minmax", min_in=0, max_in=2000)
    return im, im_sitk

def imshow(
    image: Tensor,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> None:
    r"""Render image data in last two tensor dimensions using matplotlib.pyplot.imshow().

    Args:
        image: Image tensor of shape ``(..., H, W)``.
        ax: Figure axes to render the image in. If ``None``, a new figure is created.
        label: Image label to display in the axes title.
        kwargs: Keyword arguments to pass on to ``matplotlib.pyplot.imshow()``.
            When ``ax`` is ``None``, can contain ``figsize`` to specify the size of
            the figure created for displaying the image.

    """
    if ax is None:
        figsize = kwargs.pop("figsize", (4, 4))
        _, ax = plt.subplots(figsize=figsize)
    kwargs["cmap"] = kwargs.get("cmap", "gray")
    ax.imshow(image.reshape((-1,) + image.shape[-2:])[0].cpu().numpy(), **kwargs)
    if label:
        ax.set_title(label, fontsize=16, y=1.04)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def load_nifty(image_path, image_size=None, reorient=False):

    image_sitk = sitk.ReadImage(image_path)
    if reorient:
        image = sitk.DICOMOrient(image_sitk, 'RPI')
    image = sitk.GetArrayFromImage(image)
    if image_size:
        image = crop_and_pad(image.reshape(1, *image.shape), new_size=image_size)

    image = torch.tensor(image)

    return image, image_sitk


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args.data_dir)
