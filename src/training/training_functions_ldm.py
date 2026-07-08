""" Training functions for the different models. from https://github.com/Warvito/generative_brain.git"""
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from util import log_ldm_sample_unconditioned, denoise, embed_condition, sample_image
from transformers import CLIPTokenizer
import random
import numpy as np
from dataset import setup_dataset, UKBBDataModule
import sys
from skimage.measure import block_reduce
sys.path.append('../testing')
from sample_images import sample_im
from util import warp
from dataset import load_nifty
from util import flow_2_image, normalise, visualise_deformation
sys.path.append("../vmorph/")

from model.loss import l2reg_loss, bending_energy_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    train_scheduler: nn.Module,
    inference_scheduler: nn.Module,
    text_encoder,
    tokenizer,
    atlas,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    alpha: float,
    regularization: str,
    scale_factor: float = 1.0,
    vmorph: nn.Module = None,
    gradient_decoder: bool = False,
) -> float:
    scaler = GradScaler()

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
            model=model,
            stage1=stage1,
            train_scheduler=train_scheduler,
            inference_scheduler=inference_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            atlas=atlas,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            alpha=alpha,
            regularization=regularization,
            scale_factor=scale_factor,
            vmorph=vmorph,
            gradient_decoder=gradient_decoder
        )

        val_loss = eval_ldm(
            model=model,
            stage1=stage1,
            scheduler=train_scheduler,
            text_encoder=text_encoder,
            loader=val_loader,
            device=device,
            step=len(train_loader) * epoch,
            writer=writer_val,
            sample=False,
            scale_factor=scale_factor,
        )

        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
        print_gpu_memory_report()

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "diffusion": model.state_dict(),
            "stage1_decoder": stage1.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

        if val_loss <= best_loss:
            print(f"New best val loss {val_loss}")
            best_loss = val_loss
            torch.save(model.state_dict(), str(run_dir / "best_model.pth"))
            torch.save(checkpoint, str(run_dir / "best_checkpoint.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))


    return val_loss

def my_loss(outputs, targets):
    loss = []
    for o, t in zip(outputs, targets):
        loss.append(torch.mean((o - t)**2).item())
    # loss = loss/len(outputs)
    return torch.mean(torch.Tensor(loss))


def distance_to_images(stage1,
                       diffusion,
                       vmorph,
                       text_encoder,
                       scheduler,
                       train_loader,
                       context,
                       atlas,
                       spatial_shape,
                       writer,
                       step,
                       gradient_decoder,
                       device='cuda'):
    # generate template
    atlas_disp = sample_image(diffusion,
                                stage1,
                                context[0],
                                scheduler,
                                text_encoder,
                                train_loader.dataset.tokenizer,
                                scale_factor=0.3,
                                spatial_shape=spatial_shape,
                                device=device,
                                context_type=train_loader.dataset.context,
                                gradient_decoder=gradient_decoder,)

    # load atlas

    if train_loader.dataset.downsample:
        print('downsample')
        atlas = torch.Tensor(block_reduce(atlas, block_size=2)).squeeze().to(device)
        atlas=normalise(atlas)
    with autocast(enabled=True):
        # generate conditional atlas
        gen_template = warp(atlas.unsqueeze(0), atlas_disp)
    if step %50 == 0:
        fig = visualise_deformation(im1=atlas, warped=gen_template, disp=atlas_disp, cfs_size=context[0], type='csf_volume')
        writer.add_figure("generated disp", fig, step)

    gen_template = gen_template.to(device)
    # select subgroup of images
    dl_reg = train_loader.dataset.__get_neighbourhood__(context=context[0], batch_size=1, neighborhood_size=20)
    displacements = []
    avg_disp = torch.zeros_like(atlas_disp)
    for x in dl_reg:
        with autocast(enabled=True):
            _, disp = vmorph(gen_template, x[0].to(device))
        avg_disp += disp
    mean_disp = avg_disp/len(dl_reg)
    return F.mse_loss(mean_disp, torch.zeros_like(mean_disp)), atlas_disp

def train_epoch_ldm(
    model: nn.Module,
    stage1: nn.Module,
    train_scheduler: nn.Module,
    inference_scheduler: nn.Module,
    text_encoder,
    tokenizer,
    atlas,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    alpha: float,
    regularization: str,
    scale_factor: float = 1.0,
    vmorph: nn.Module = None,
    gradient_decoder: bool = False,

) -> None:
    model.train()
    stage1.eval()

    loss_diffusion = 0

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        if text_encoder is not None:
            images, conditions, context = x[0].to(device), x[1].to(device), x[2]#.unsqueeze(1).to(torch.long)
        else:
            images = x.to(device)

        timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                # encode the images into the latent space
                e = stage1.encode_stage_2_inputs(images) * scale_factor

            # add random noise to the image given a random timestep
            noise = torch.randn_like(e).to(device)
            noisy_e = train_scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            if text_encoder is not None:
                # embed the tokenized conditions
                prompt_embeds = text_encoder(conditions.squeeze(1))
                prompt_embeds = prompt_embeds[0]
                # predict added noise to the image
                noise_pred = model(x=noisy_e, timesteps=timesteps, context=prompt_embeds)
            else:
                # predict added noise to the image
                noise_pred = model(x=noisy_e, timesteps=timesteps)

            if train_scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = train_scheduler.get_velocity(e, noise, timesteps)
            elif train_scheduler.prediction_type == "epsilon":
                target = noise

            loss_diffusion = F.mse_loss(noise_pred.float(), target.float())

        loss_reg, gen_disp  = distance_to_images(stage1,
                                       model,
                                       vmorph,
                                       text_encoder,
                                       inference_scheduler,
                                       loader,
                                       context,
                                       atlas=atlas,
                                       spatial_shape=tuple(e.shape[1:]),
                                       gradient_decoder=gradient_decoder,
                                       device=device,
                                       writer=writer,
                                       step=step)
        if regularization == 'l2':
            reg_disp = alpha*l2reg_loss(gen_disp)
        else:
            reg_disp =alpha*bending_energy_loss(gen_disp)
        loss = loss_diffusion + loss_reg + reg_disp
        # brackward pass + gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log to tensorboard
        writer.add_scalar(f"loss", loss.item(), epoch * len(loader) + step)
        writer.add_scalar(f"diffusion loss", loss_diffusion.item(), epoch * len(loader) + step)
        writer.add_scalar(f"reg loss", loss_reg.item(), epoch * len(loader) + step)
        pbar.set_postfix({"epoch": epoch, "loss": f"{loss.item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})


@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    sample: bool = False,
    scale_factor: float = 1.0,
) -> float:
    model.eval()
    total_losses = OrderedDict()

    stage1 = stage1.to(device)

    for x in tqdm(loader):
        if text_encoder is not None:
            # images, conditions = x[0].to(device), x[1].to(device)
            images, conditions, cfs_size = x[0].to(device), x[1].to(device), x[2]
        else:
            images = x.to(device)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(enabled=True):
            # encode the images into the latent space
            e = stage1.encode_stage_2_inputs(images) * scale_factor
            # add random noise to the image given a random timestep
            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            if text_encoder is not None:
                prompt_embeds = text_encoder(conditions.squeeze(1))
                prompt_embeds = prompt_embeds[0]
                noise_pred = model(x=noisy_e, timesteps=timesteps, context=prompt_embeds)
            else:
                noise_pred = model(x=noisy_e, timesteps=timesteps)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    return total_losses["loss"]
