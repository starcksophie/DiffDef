""" Script to generate sample images from the diffusion model.from https://github.com/Warvito/generative_brain.git

In the generation of the images, the script is using a DDIM scheduler.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
import sys
sys.path.append('../training')
from dataset import load_sitk, save_sitk

def embed_inputs(cfs_size, tokenizer, text_encoder, device):
    prompt = f"T1-weighted image of a brain with a ventricle size of {cfs_size}."
    print(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.squeeze(1).to(device))
    prompt_embeds = prompt_embeds[0].to(device)
    return prompt_embeds

def sample_im(cfs_size, stage1, diffusion, tokenizer, text_encoder, scheduler, device, training_mode=False):
    prompt_embeds =embed_inputs(cfs_size, tokenizer, text_encoder, device)
    spatial_shape = (3, 20, 28, 20)
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    latents = []

    with torch.set_grad_enabled(False):
        for t in tqdm(scheduler.timesteps, ncols=70):
            noise_pred = diffusion(x=latent, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
            latent, _ = scheduler.step(noise_pred, t, latent)
            latents.append(latent)
    stage1.eval()

    latent = latents[-1]
    stage1 = stage1.to('cpu')
    with torch.set_grad_enabled(training_mode):
        x_hat = stage1.decode(latent.cpu() / 0.3)
    stage1 = stage1.to('cuda')
    return x_hat

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.")
    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--stage1_config_file_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--reference_path", help="Path to the reference image.")
    parser.add_argument("--prompt", help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--num_inference_steps", type=int, help="")
    parser.add_argument("--csf_size", type=float, help="")

    args = parser.parse_args()
    return args

def main(args):
    print_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

    config = OmegaConf.load(args.stage1_config_file_path)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(args.diffusion_config_file_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(args.diffusion_path))
    diffusion.to(device)
    diffusion.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)

    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

    set_determinism(seed=2)
    sample = sample_im(args.cfs_size, stage1, diffusion, tokenizer, text_encoder, scheduler, device).squeeze().detach()

    sample = np.clip(sample.cpu().numpy(), 0, 1)
    sample = (sample * 255).astype(np.uint8)
    ref_sitk = load_sitk(args.reference_path)

    save_sitk(output_dir / f"sample_{args.csf_size}.nii.gz", sample, ref_sitk)


if __name__ == "__main__":
    args = parse_args()
    main(args)
