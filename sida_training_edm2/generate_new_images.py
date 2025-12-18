import os
import re
import logging
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib

from torch_utils import distributed as dist
from sida_training_edm2.vae_edm2_utils import load_sd_vae
from training.simple_dataset import ImageDataset

LOGGER = logging.getLogger(name='__main__')

# ----------------------------------------------------------------------------
# One-step generator that allows specifying a different random seed for each generated sample


### Constants
raw_mean = np.array([5.81, 3.25, 0.12, -2.15], dtype=np.float32)  # Assumed mean of the raw latents.
raw_std = np.array([4.17, 4.62, 3.71, 3.28],
                   dtype=np.float32)  # Assumed standard deviation of the raw latents.
final_mean = np.float32(0)  # Desired mean of the final latents.
final_std = np.float32(0.5)  # Desired standard deviation of the final latents.
SCALE = final_std / raw_std
BIAS = final_mean - raw_mean * SCALE


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in
                           seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in
                            self.generators])


def run_generator(G, z, c, init_sigma, opts):
    with torch.no_grad():
        # Ensure all necessary attributes are in opts and are of tensor type
        scale_ = torch.tensor(opts.scale,
                              dtype=torch.float32,
                              device=z.device) if hasattr(opts, 'scale') else torch.tensor(1.0,
                                                                                           device=z.device)
        bias_ = torch.tensor(opts.bias,
                             dtype=torch.float32,
                             device=z.device) if hasattr(opts, 'bias') else torch.tensor(0.0,
                                                                                         device=z.device)
        vae = opts.vae if hasattr(opts, 'vae') else None

        # Initialize sigma tensor and pass through G
        init_sigma_tensor = init_sigma * torch.ones(z.shape[0], 1, 1, 1, device=z.device)
        img = G(z, init_sigma_tensor, c, augment_labels=torch.zeros(z.shape[0], 9, device=z.device))

        if vae is not None:
            vae.to(z.device)  # Ensure VAE is on the same device as the input tensor

            img = (img - bias_.reshape(1, -1, 1, 1)) / scale_.reshape(1, -1, 1, 1)

            img = vae.decode(img).sample  # Ensure `.sample` is compatible with `vae.decode`

        # downsample to 224x224
        if opts.downsample:
            img = torch.nn.functional.interpolate(img, size=(224, 224),
                                                  mode='bilinear', align_corners=False)

        # Final scaling to uint8 for image representation
        img = img.clamp(0, 1).mul(255).to(torch.uint8)
        # print(img.shape)  # For debugging purposes

    return img


def generate_new_images(net,
                        num_each,
                        sigma_G=2.5,
                        max_batch_size=16,
                        device=torch.device('cuda')
                        ):
    random_start = torch.randint(0, 2 ** 31, (1,)).item()
    seeds = torch.arange(num_each) + random_start
    num_batches = ((len(seeds) - 1) // (
            max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    net.eval().to(device)
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # initialize a new torch dataset proj
    dataset = ImageDataset(net.label_dim)

    pretrained_vae_model_name_or_path = 'stabilityai/sd-vae-ft-mse'
    vae = load_sd_vae(pretrained_vae_model_name_or_path, device=device)
    opts = dnnlib.EasyDict()
    opts.vae = vae
    opts.scale = SCALE
    opts.bias = BIAS
    opts.downsample = True

    for label in range(net.label_dim):
        LOGGER.info(f"Generating images for label {label}")
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            rnd = StackedRandomGenerator(device, batch_seeds)

            latents = rnd.randn(
                [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
                device=device)
            class_labels = None
            # print(net.label_dim)
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[
                    rnd.randint(net.label_dim, size=[batch_size], device=device)]
            if label is not None:
                class_labels[:, :] = 0
                class_labels[:, label] = 1

            # print(f"latents shape: {latents.shape}") # torch.Size([16, 3, 32, 32])
            # print(f"class_labels shape: {class_labels.shape}") # class_labels shape: torch.Size([16, 10])
            # print(f"sigma_G: {sigma_G}") # sigma_G: 2.5

            sigma = sigma_G * torch.ones([batch_size, 1, 1, 1], device=device)
            images = net(sigma_G * latents.to(torch.float32), sigma,
                         class_labels)  # .to(torch.float32)

            # images = run_generator(net,
            #                        z=sigma_G * latents.to(torch.float32),
            #                        c=class_labels,
            #                        init_sigma=sigma_G,
            #                        opts=opts)

            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()

            for i, img in enumerate(images_np):
                dataset.append(img, label)

    return dataset
