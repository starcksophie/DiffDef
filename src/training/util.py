from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import start_run
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
from transformers import CLIPTokenizer
import torch.nn.functional as F
import sys
from dataset import load_nifty
# sys.path.append('../src/neurite/neurite/py')
# from plot import flow

# ----------------------------------------------------------------------------------------------------------------------
# LOGS from from https://github.com/Warvito/generative_brain.git
# ----------------------------------------------------------------------------------------------------------------------
def recursive_items(dictionary, prefix=""):
    for key, value in dictionary.items():
        if type(value) in [dict, DictConfig]:
            yield from recursive_items(value, prefix=str(key) if prefix == "" else f"{prefix}.{str(key)}")
        else:
            yield (str(key) if prefix == "" else f"{prefix}.{str(key)}", value)


def log_mlflow(
    model,
    config,
    args,
    experiment: str,
    run_dir: Path,
    val_loss: float,
):
    """Log model and performance on Mlflow system"""
    config = {**OmegaConf.to_container(config), **vars(args)}
    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, str(value))

        mlflow.log_artifacts(str(run_dir / "train"), artifact_path="events_train")
        mlflow.log_artifacts(str(run_dir / "val"), artifact_path="events_val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model
        mlflow.pytorch.log_model(raw_model, "final_model")


def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
):
    img = img.squeeze()
    recons = recons.squeeze()
    shape = img.shape
    img_npy_0 = np.clip(a=img[shape[0]//2].cpu().numpy(), a_min=0, a_max=1)
    img_npy_1 = np.clip(a=img[:, shape[1]//2].cpu().numpy(), a_min=0, a_max=1)
    img_npy_2 = np.clip(a=img[:, :, shape[2]//2].cpu().numpy(), a_min=0, a_max=1)
    shape = recons.shape
    recons_npy_0 = np.clip(a=recons[shape[0]//2].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_1 = np.clip(a=recons[:, shape[1]//2].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_2 = np.clip(a=recons[:, :, shape[2]//2].cpu().numpy(), a_min=0, a_max=1)


    fig, ax = plt.subplots(3, 2, dpi=300, figsize=(5, 10))
    ax[0][0].imshow(np.rot90(img_npy_0, k=2), cmap="gray")
    ax[1][0].imshow(np.rot90(img_npy_1, k=2), cmap="gray")
    ax[2][0].imshow(np.rot90(img_npy_2, k=2), cmap="gray")

    ax[0][1].imshow(np.rot90(recons_npy_0, k=2), cmap="gray")
    ax[1][1].imshow(np.rot90(recons_npy_1, k=2), cmap="gray")
    ax[2][1].imshow(np.rot90(recons_npy_2, k=2), cmap="gray")

    for i in [0, 1]:
        ax[0][i].axis("off")
        ax[1][i].axis("off")
        ax[2][i].axis("off")
    plt.savefig('test.png')
    return fig


def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
) -> None:
    fig = get_figure(
        image,
        reconstruction,
    )
    writer.add_figure(title, fig, step)


def embed_condition(context, text_encoder, tokenizer, device, context_type='csf_volume'):
    # tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
    if not context:
        # select a random size for the ventricle
        c = torch.Tensor([random.choice(np.arange(0.1, 1, 0.1))])
    else:
        c = torch.Tensor([context])
    if context_type == 'csf_volume':
        prompt = f"T1-weighted image of a brain with a ventricle size of {c[0]:.1f}."
    else:
        prompt = f"T1-weighted image of a {int(100*c[0])} years old brain."
    # embed the inputs
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.squeeze(1).to(device))
    prompt_embeds = prompt_embeds[0]
    return prompt, prompt_embeds

def denoise(latent, model, scheduler, device, text_encoder=None, prompt_embeds=None):
    for t in scheduler.timesteps:
        if text_encoder is not None:
            noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
        else:
            noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device))
        latent, _ = scheduler.step(noise_pred, t, latent)
    return latent

def sample_image(model, stage1, condition, scheduler, text_encoder, tokenizer, scale_factor, spatial_shape, device, context_type, gradient_decoder=False,):
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    _, prompt_embeds = embed_condition(condition, text_encoder, tokenizer, device, context_type=context_type)

    # diffusion process
    with torch.no_grad():
        latent = denoise(latent, model, scheduler, device, text_encoder, prompt_embeds)
    # with torch.set_grad_enabled(gradient_decoder):
    x_hat = stage1.decode(latent / scale_factor)
    return x_hat

@torch.no_grad()
def log_ldm_sample_unconditioned(
    model: nn.Module,
    stage1: nn.Module,
    text_encoder,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
    condition: float  = None,
    write: bool = True,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    prompt, prompt_embeds = embed_condition(condition, text_encoder, device)
    if write:
        print('sampling..')

    # diffusion process
    latent = denoise(latent, model, scheduler, device, text_encoder, prompt_embeds)

    x_hat = stage1.decode(latent / scale_factor)

    if write:
        x = x_hat[0].squeeze()
        img_npy_0 = np.clip(a=x[x.shape[0]//2].cpu().numpy(), a_min=0, a_max=1)
        img_npy_1 = np.clip(a=x[:, x.shape[1]//2].cpu().numpy(), a_min=0, a_max=1)
        img_npy_2 = np.clip(a=x[:, :, x.shape[2]//2].cpu().numpy(), a_min=0, a_max=1)


        fig, ax = plt.subplots(1, 3, dpi=300)
        ax[0].imshow(np.rot90(img_npy_0, k=2), cmap="gray")
        ax[1].imshow(np.rot90(img_npy_1, k=2), cmap="gray")
        ax[2].imshow(np.rot90(img_npy_2, k=2), cmap="gray")

        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        plt.suptitle(prompt)
        writer.add_figure("SAMPLE", fig, step)
    return x_hat


def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors

def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)], indexing='ij')
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)

def flow_2_image(d, show=False):
    d = d[::max(1, d.shape[0] // 40), ::max(1, d.shape[0] // 40)]
    d = d.numpy() if isinstance(d, torch.Tensor) else d
    fig, _ = flow((np.rot90([d], k=2)), show=show, img_indexing=False)
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(*reversed(canvas.get_width_height()), 3)
    plt.close(fig)
    return image
    # return fig

def normalise(x):
    return (x - x.min()) / (x.max() - x.min())

def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def visualise_deformation(im1, warped, disp, cfs_size, type='age'):
    if gt:
        s = 4
    else:
        s = 3
    fig, ax = plt.subplots(1, s, figsize=(20, 10))

    a = im1.detach().cpu().squeeze().numpy()
    w = warped.detach().cpu().squeeze().numpy()

    ii = a.shape[0]//2

    ind = 0

    ax[ind].imshow(a[ii], cmap='gray')
    ax[ind].set_title(f'MNI')
    ax[ind+1].imshow(w[ii], cmap='gray')
    ax[ind+1].set_title(f'Warped {cfs_size}')

    plot_warped_grid(ax[ind+2], disp[0, :2, ii].detach().cpu().numpy(), None, interval=3, fontsize=20)
    for i,axx in enumerate(ax):
        axx.axis('off')
    return fig