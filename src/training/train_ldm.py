""" Training script for the diffusion model in the latent space of the pretraine AEKL model from https://github.com/Warvito/generative_brain.git """
import argparse
import warnings
from pathlib import Path
from generative.networks.nets import DiffusionModelUNet
import mlflow.pytorch
import torch
import torch.optim as optim
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions_ldm import train_ldm
from util import log_mlflow
from dataset import setup_dataset
from monai.networks.blocks import Convolution
from transformers import CLIPTextModel, CLIPTokenizer
# from monai.networks.blocks import Convolution
from mapper import Mapper
from torch import nn
import sys
import os
# sys.path.append(os.path.dirname(os.path.realpath('__file__')).split('Diffatlas')[0]+'Diffatlas/src/vmorph/')
sys.path.append("../vmorph/")
from model.lightning import LightningDLReg
from itertools import chain


warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--atlas_path",help="Location of MNI reference atlas.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--scale_factor", default=0.3, type=float, help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--context", default=True, help="Whether the model takes text as an additional input")
    parser.add_argument("--pretrained_diff", default=False, help="Whether the model is pretrained or not")

    args = parser.parse_args()
    return args

def setup_output_dirs(config):
    output_dir = Path(config.paths.work_dir+"ldm_runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / config.paths.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    cache_dir = output_dir / "cached_data_diffusion"
    cache_dir.mkdir(exist_ok=True)

    return run_dir, resume

    print(f"Run directory: {str(run_dir)}")

def main(args):
    set_determinism(seed=args.seed)
    print_config()

    config = OmegaConf.load(args.config_file)

    run_dir, resume  = setup_output_dirs(config)

    # tensorboard log directories
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    device = torch.device("cuda")

    train_loader, val_loader = setup_dataset(config.paths.training_ids, config.paths.validation_ids, args.batch_size, config.ldm.context, image_size=(160, 224, 160), downsample=False)
    atlas = = load_nifty(args.atlas_path, (160, 224, 160))
    ### LOADING PRETRAINED AE MODEL
    print(f"Loading Stage 1 from {config.paths.aekl_path}")
    stage1 = mlflow.pytorch.load_model(config.paths.aekl_path)
    stage1.decoder.blocks[-1] = Convolution(
                    spatial_dims=3,
                    in_channels=32,
                    out_channels=3,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                    )
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad = False

    if config.ldm.trainable_decoder:
        stage1.decoder.train()
        for param in stage1.decoder.parameters():
            param.requires_grad = True

    for param in stage1.decoder.blocks[-1].parameters():
        param.requires_grad = True
    stage1.decoder.blocks[-1].train()

    print("Creating model...")
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    train_scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    val_scheduler = DDIMScheduler(**config["ldm"].get("scheduler", dict()))
    num_inference_steps=20
    val_scheduler.set_timesteps(num_inference_steps)

    if args.context:
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
        text_encoder = text_encoder.to(device)
    else:
        text_encoder = None

    ### LOADING VOXELMORPH MODEL
    vmorph = LightningDLReg.load_from_checkpoint(config.paths.vmorph_path)
    vmorph = vmorph.to(device=device)
    vmorph.eval()

    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    vmorph = vmorph.to(device)

    if config.ldm.trainable_decoder:
        params = chain(diffusion.parameters(), stage1.decoder.parameters())
    else:
        params = diffusion.parameters()

    optimizer = optim.AdamW(params, lr=config["ldm"]["base_lr"])
    if args.pretrained_diff:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0

    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        diffusion.load_state_dict(checkpoint["diffusion"])
        stage1.decoder.load_state_dict(checkpoint["stage1_decoder"])
        # Issue loading optimizer https://github.com/pytorch/pytorch/issues/2830
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print(f"No checkpoint found.")

    # Train model
    print(f"Starting Training")
    val_loss = train_ldm(
        model=diffusion,
        stage1=stage1,
        train_scheduler=train_scheduler,
        inference_scheduler=val_scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        atlas=atlas,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        alpha=config["ldm"]["displacement"]["alpha"],
        regularization=config["ldm"]["displacement"]["regularization"],
        scale_factor=args.scale_factor,
        vmorph=vmorph,
        gradient_decoder=config.ldm.trainable_decoder,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
