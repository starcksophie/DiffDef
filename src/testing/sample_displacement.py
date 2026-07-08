import numpy as np
import torch
import os
from omegaconf import OmegaConf
import sys
import matplotlib.pyplot as plt
sys.path.append('../training')
sys.path.append('../testing')
from util import warp, plot_warped_grid,  denoise, warp, embed_condition
from dataset import load_nifty, save_sitk
import mlflow.pytorch
from monai.networks.blocks import Convolution
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import SimpleITK as sitk

def center_slice(im):
    im=torch.Tensor(im).squeeze()
    return im[im.shape[0]//2].cpu().numpy()

def load_diffusion_models(stage1_path, diffusion_path, config, device):
    stage1 =  mlflow.pytorch.load_model(stage1_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(diffusion_path)['diffusion'])
    stage1.decoder.blocks[-1] = Convolution(
                        spatial_dims=3,
                        in_channels=32,
                        out_channels=3,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                        )
    stage1.decoder.load_state_dict(torch.load(diffusion_path)["stage1_decoder"])
    stage1.eval()
    diffusion.eval()
    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    return stage1, diffusion

def setup_diffusion(config, stage1_path, diffusion_path, device):
    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
    num_inference_steps=20
    scheduler.set_timesteps(num_inference_steps)
    # models
    stage1, diffusion = load_diffusion_models(stage1_path, diffusion_path, config, device)
    return stage1, diffusion, scheduler, tokenizer, text_encoder

@torch.no_grad()
def sample_template_disp(cond, context_type, atlas, diffusion, stage1, scheduler, tokenizer, text_encoder, device):
    promp, prompt_embeds =embed_condition(cond, tokenizer=tokenizer, text_encoder=text_encoder.to(device), device=device, context_type=context_type)
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)
    stage1.eval()
    # stage1 = stage1.to('cpu')
    # with torch.set_grad_enabled(False):÷
    with torch.no_grad():
        latent = denoise(latent, diffusion, scheduler, device, text_encoder, prompt_embeds)
        gen_disp = stage1.decode(latent / 0.3)
    gen_template = warp(atlas.unsqueeze(0), gen_disp)
    return gen_template, gen_disp


def sample(context_type, atlas, diffusion, stage1, scheduler, tokenizer, text_encoder, device):
    if context_type =='age':
        sampling_range = np.arange(0.5, 0.8, 0.05, dtype=float)
    elif context_type == 'csf_volume':
        sampling_range = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    templates, disps = {}, {}
    for cond in sampling_range:
            templates[cond], disps[cond] = sample_template_disp(cond, context_type, atlas, diffusion, stage1, scheduler, tokenizer, text_encoder, device)
    return templates, disps, sampling_range

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--atlas_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--path_out", help="Path to the .pth model from the diffusion model.")
    device='cuda:0'

    config = OmegaConf.load(diffusion_config_file_path)
    stage1, diffusion, scheduler, tokenizer, text_encoder = setup_diffusion(config, config.paths.aekl_path, diffusion_path, device)
    atlas = load_nifty(atlas_path,  image_size=(160, 224, 160))
    sitk_arr = sitk.ReadImage(atlas_path)
    sitk_arr = sitk.DICOMOrient(sitk_arr, 'RPI')
    spatial_shape=(3, 20, 28, 20)
    CONTEXT = 'csf_volume'
    # CONTEXT = 'age'
    print('SAMPLING IMAGES')
    for fold in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
        templates, disps, sampling_range = sample(CONTEXT, atlas, diffusion, stage1, scheduler, tokenizer, text_encoder, device)
        for key, val in templates.items():
            if CONTEXT =='age':
                key = int(100*key)
            save_sitk(os.path.join(path_out, fold, diffusion_path.split('/')[-2]+f'_{CONTEXT}_{key}.nii.gz'), templates[key].squeeze(), sitk_arr)
            save_sitk(os.path.join(path_out, fold, diffusion_path.split('/')[-2]+f'_{CONTEXT}_{int(100*key)}.nii.gz'), templates[key].squeeze(), sitk_arr)
            torch.save(disps[key], os.path.join(path_out, fold, diffusion_path.split('/')[-2]+f'_{CONTEXT}_{key}_disp.pt'))