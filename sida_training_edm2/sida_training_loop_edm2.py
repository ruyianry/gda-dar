# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step"."""

"""Main training loop."""

import os
import logging
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats

from functools import partial

from torch_utils import misc
from edm2 import misc as misc2

# from metrics import sid_metric_main as metric_main
from metrics_edm2 import sid_metric_main_edm2 as metric_main
from sida_training_edm2.vae_edm2_utils import load_sd_vae, \
    vae_sampler_edm2  # ,vae_edm2_encode_latents, vae_edm2_decode,  #, vae_edm2_decode

LOGGER = logging.getLogger(name='__main__')

####
# raw_mean    = [5.81, 3.25, 0.12, -2.15]    # Assumed mean of the raw latents.
# raw_std     = [4.17, 4.62, 3.71, 3.28]     # Assumed standard deviation of the raw latents.
# final_mean  = 0                            # Desired mean of the final latents.
# final_std   = 0.5                          # Desired standard deviation of the final latents.
# scale = np.float32(final_std) / np.float32(raw_std)
# bias = np.float32(final_mean) - np.float32(raw_mean) * scale

raw_mean = np.array([5.81, 3.25, 0.12, -2.15], dtype=np.float32)  # Assumed mean of the raw latents.
raw_std = np.array([4.17, 4.62, 3.71, 3.28],
                   dtype=np.float32)  # Assumed standard deviation of the raw latents.
final_mean = np.float32(0)  # Desired mean of the final latents.
final_std = np.float32(0.5)  # Desired standard deviation of the final latents.
scale = final_std / raw_std
bias = final_mean - raw_mean * scale


# def vae_edm2_encode_latents(scale,bias,x): # raw latents => final latents
#     mean, std = x.to(torch.float32).chunk(2, dim=1)
#     x = mean + torch.randn_like(mean) * std
#     x = x * misc2.const_like(x, scale).reshape(1, -1, 1, 1)
#     x = x + misc2.const_like(x, bias).reshape(1, -1, 1, 1)
#     return x

# def vae_edm2_decode(vae,scale,bias, x): # final latents => raw pixels
#     vae.to(x.device)
#     x = x.to(torch.float32)
#     x = x - misc2.const_like(x, bias).reshape(1, -1, 1, 1)
#     x = x / misc2.const_like(x, scale).reshape(1, -1, 1, 1)
#     x = vae.decode(x).sample
#     #x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
#     x = x.clamp(0, 1).mul(255).to(torch.uint8)
#     return x


# def vae_edm2_encode_latents(scale, bias, x): # raw latents => final latents
#     mean, std = x.to(torch.float32).chunk(2, dim=1)
#     x = mean + torch.randn_like(mean) * std
#     x = x * scale.reshape(1, -1, 1, 1).to(x.device)
#     x = x + bias.reshape(1, -1, 1, 1).to(x.device)
#     return x

# def vae_edm2_decode(vae, scale, bias, x): # final latents => raw pixels
#     vae.to(x.device)  # Ensure VAE is on the same device as the input tensor
#     x = x.to(torch.float32)
#     x = x - bias.reshape(1, -1, 1, 1).to(x.device)
#     x = x / scale.reshape(1, -1, 1, 1).to(x.device)
#     x = vae.decode(x).sample  # Ensure `vae.decode(x)` returns a compatible object with `.sample`
#     x = x.clamp(0, 1).mul(255).to(torch.uint8)
#     return x


def vae_edm2_encode_latents(scale, bias, x):  # raw latents => final latents
    # Ensure scale and bias are PyTorch tensors
    scale = torch.tensor(scale, dtype=torch.float32, device=x.device)
    bias = torch.tensor(bias, dtype=torch.float32, device=x.device)

    mean, std = x.to(torch.float32).chunk(2, dim=1)
    x = mean + torch.randn_like(mean) * std
    x = x * scale.reshape(1, -1, 1, 1)
    x = x + bias.reshape(1, -1, 1, 1)
    return x


# def vae_edm2_decode(vae, scale, bias, x): # final latents => raw pixels
#     # Convert scale and bias to tensors if they are not already
#     scale = torch.tensor(scale, dtype=torch.float32, device=x.device)
#     bias = torch.tensor(bias, dtype=torch.float32, device=x.device)

#     vae.to(x.device)  # Ensure VAE is on the same device as the input tensor
#     x = x.to(torch.float32)
#     x = x - bias.reshape(1, -1, 1, 1)
#     x = x / scale.reshape(1, -1, 1, 1)
#     x = vae.decode(x).sample  # Ensure `vae.decode(x)` returns a compatible object with `.sample`
#     x = x.clamp(0, 1).mul(255).to(torch.uint8)
#     return x


def vae_edm2_decode(vae, scale, bias, x):  # final latents => raw pixels
    # Check if scale and bias are already tensors to avoid redundant conversions
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    if not isinstance(bias, torch.Tensor):
        bias = torch.as_tensor(bias, dtype=torch.float32, device=x.device)

    # Ensure VAE is on the correct device and in evaluation mode
    vae = vae.to(x.device).eval()

    # Normalize the input `x` with scale and bias
    x = (x.to(torch.float32) - bias.view(1, -1, 1, 1)) / scale.view(1, -1, 1, 1)

    # Decode and process the output image
    with torch.no_grad():
        x = vae.decode(x).sample  # Ensure the output has `.sample()` if required

    # Clamp and convert the image to uint8 in a single step
    return (x.clamp(0, 1) * 255).to(torch.uint8)


#######


import itertools


def setup_snapshot_image_grid(training_set,
                              training_loader,
                              label_dim,
                              has_labels,
                              random_seed=0,
                              max_iterations=2000):
    rnd = np.random.RandomState(random_seed)
    # gw = np.clip(3840 // training_set.image_shape[2], 7, 32)
    # gh = np.clip(2160 // training_set.image_shape[1], 4, 32)

    # gw = np.clip(3840 // 512, 7, 32)
    # gh = np.clip(2160 // 512, 4, 32)
    gw = np.clip(2048 // 512, 2, 32)
    gh = np.clip(2048 // 512, 2, 32)

    # No labels => show random subset of training samples.
    if not has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]

        # Set a maximum number of iterations

        for label in range(label_dim):
            label_groups[label] = []

        for i, (image, label, idx) in enumerate(training_loader):
            label = np.argmax(label)
            # to int
            label = int(label)
            label_groups[label].append(i)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]

            # if inidices is empty, skip
            if len(indices) == 0:
                continue

            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, _ = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------
# def setup_snapshot_image_grid(training_set, random_seed=0):
#     rnd = np.random.RandomState(random_seed)
#     #gw = np.clip(3840 // training_set.image_shape[2], 7, 32)
#     #gh = np.clip(2160 // training_set.image_shape[1], 4, 32)

#     #gw = np.clip(3840 // 512, 7, 32)
#     #gh = np.clip(2160 // 512, 4, 32)
#     gw = np.clip(1024 // 512, 2, 32)
#     gh = np.clip(1024 // 512, 2, 32)

#     # No labels => show random subset of training samples.
#     if not training_set.has_labels:
#         all_indices = list(range(len(training_set)))
#         rnd.shuffle(all_indices)
#         grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

#     else:
#         # Group training samples by label.
#         label_groups = dict() # label => [idx, ...]
#         for idx in range(len(training_set)):
#             label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
#             if label not in label_groups:
#                 label_groups[label] = []
#             label_groups[label].append(idx)

#         # Reorder.
#         label_order = sorted(label_groups.keys())
#         for label in label_order:
#             rnd.shuffle(label_groups[label])

#         # Organize into grid.
#         grid_indices = []
#         for y in range(gh):
#             label = label_order[y % len(label_order)]
#             indices = label_groups[label]
#             grid_indices += [indices[x % len(indices)] for x in range(gw)]
#             label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

#     # Load data.
#     images, labels = zip(*[training_set[i] for i in grid_indices])
#     return (gw, gh), np.stack(images), np.stack(labels)


def setup_snapshot_image_grid_decode(training_set,
                                     training_loader,
                                     label_dim,
                                     has_labels,
                                     random_seed=0,
                                     encoder=None,
                                     device=None,
                                     max_iterations=2000):
    rnd = np.random.RandomState(random_seed)
    # gw = np.clip(3840 // training_set.image_shape[2], 7, 32)
    # gh = np.clip(2160 // training_set.image_shape[1], 4, 32)

    # gw = np.clip(3840 // 512, 7, 32)
    # gh = np.clip(2160 // 512, 4, 32)
    gw = np.clip(2048 // 512, 2, 32)
    gh = np.clip(2048 // 512, 2, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for label in range(label_dim):
            label_groups[label] = []

        for i, (image, label, idx) in enumerate(training_loader):
            label = np.argmax(label)
            # to int
            label = int(label)
            label_groups[label].append(i)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]

            # if inidices is empty, skip
            if len(indices) == 0:
                continue

            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    # images, labels = zip(*[encoder.decode(training_set[i].to(device)) for i in grid_indices])
    # images, labels = zip(*[encoder.decode(encoder.vae_edm2_encode_latents(torch.as_tensor(input).to(device).unsqueeze(0))) for input, label in (training_set[i] for i in grid_indices)])

    #     images = [encoder.decode(encoder.vae_edm2_encode_latents(torch.as_tensor(input).to(device).unsqueeze(0))).cpu() for input, _ in (training_set[i] for i in grid_indices)]
    #     labels = [label for _, label in (training_set[i] for i in grid_indices)]

    # return (gw, gh), np.stack(images), np.stack(labels)
    return (gw, gh), grid_indices  # np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------
# Helper methods

def save_image_grid(img, fname, drange, grid_size):
    # lo, hi = drange
    # img = np.asarray(img, dtype=np.float32)
    # img = (img - lo) * (255 / (hi - lo))
    # img = np.rint(img).clip(0, 255).astype(np.uint8)
    img = np.rint(img).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def save_image(img, num_channel, fname):
    assert C in [1, 3]
    if num_channel == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if num_channel == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def save_pt(pt, fname):
    torch.save(pt, fname)


def calculate_metric(metric, G, vae, scale, bias, init_sigma, dataset_kwargs, num_gpus, rank,
                     local_rank, device, data_stat, detector_url, batch_size, batch_gen,
                     dtype=torch.float16, train_sampler=False, num_steps=1):
    return metric_main.calc_metric(metric=metric, G=G, vae=vae, scale=scale, bias=bias,
                                   init_sigma=init_sigma,
                                   dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank,
                                   local_rank=local_rank, device=device, data_stat=data_stat,
                                   detector_url=detector_url, batch_size=batch_size,
                                   batch_gen=batch_gen, dtype=dtype, train_sampler=train_sampler,
                                   num_steps=num_steps)


def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')


# ----------------------------------------------------------------------------


def training_loop(
        run_dir='.',  # Output directory.
        dataset_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        network_kwargs={},  # Options for model and preconditioning.
        loss_kwargs={},  # Options for loss function.
        fake_score_optimizer_kwargs={},  # Options for fake score network optimizer.
        g_optimizer_kwargs={},  # Options for generator optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
        seed=0,  # Global random seed.
        batch_size=512,  # Total batch size for one training iteration.
        batch_gpu=None,  # Limit batch size per GPU, None = no limit.
        total_kimg=200000,  # Training duration, measured in thousands of training images.
        ema_halflife_kimg=500,
        # Half-life of the exponential moving average (EMA) of model weights.
        ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
        loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
        #
        loss_scaling_G=1,  # Loss scaling factor of G for reducing FP16 under/overflows.
        #
        kimg_per_tick=50,  # Interval of progress prints.
        snapshot_ticks=50,  # How often to save network snapshots, None = disable.
        state_dump_ticks=500,  # How often to dump training state, None = disable.
        resume_pkl=None,
        # Start from the given network snapshot for initialization, None = random initialization.
        resume_training=None,  # Resume training from the given network snapshot.
        resume_kimg=0,  # Start from the given training progress.
        alpha=1,  # loss = L2-alpha*L1
        tmax=800,  # We add noise at steps 0 to tmax, tmax <= 1000
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        device=torch.device('cuda'),
        metrics=None,
        init_sigma=None,
        data_stat=None,
        loss_scaling_G_gan=1,
        # Loss scaling factor of GAN's generator loss for reducing FP16 under/overflows.
        loss_scaling_D=1,
        # Loss scaling factor of GAN's discriminator loss for reducing FP16 under/overflows.
        sid_model=None,  # pre-distilled SiD generator
        detector_url=None,
        use_gan=True,
        save_best_and_last=True,
        encoder_kwargs={},
        generator_kwargs={},
        pretrained_vae_model_name_or_path='stabilityai/sd-vae-ft-mse',
        return_logvar=True,
        vaedevice=torch.device('cuda'),  # torch.device('cpu')

        train_dataset=None,  # training dataset
        train_dataset_ref=None,  # training dataset reference
):
    # Initialize.
    start_time = time.time()

    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    cur_nimg = resume_kimg * 1000

    # Load dataset.
    dist.print0('Loading dataset...')
    # dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_obj = train_dataset
    dataset_sampler = misc2.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(),
                                            num_replicas=dist.get_world_size(), seed=seed,
                                            start_idx=cur_nimg * 2)
    dataloader = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler,
                                             batch_size=batch_gpu, **data_loader_kwargs)
    dataloader_for_grid = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=1,
                                                      shuffle=False,
                                                      **data_loader_kwargs)
    dataset_iterator = iter(dataloader)

    ref_image, ref_label = dataset_obj[0]
    dist.print0(ref_image)
    dist.print0(ref_label)
    # interface_kwargs_generator = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])

    # vae = load_sd_vae(pretrained_vae_model_name_or_path, device, weight_dtype=dtype)

    vae = load_sd_vae(pretrained_vae_model_name_or_path, device=vaedevice)  # , weight_dtype=dtype)
    # vae = load_sd_vae(pretrained_vae_model_name_or_path, device=device)

    # encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    # encoder = encoder.to(device)
    # encoder.eval().requires_grad_(False).to(device)
    ref_image = vae_edm2_encode_latents(scale, bias,
                                        torch.as_tensor(ref_image).to(device).unsqueeze(0))

    dist.print0(ref_image)

    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=train_dataset_ref.resolution,
                            img_channels=train_dataset_ref.num_channels,
                            label_dim=train_dataset_ref.label_dim)
    # interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    # Construct the pretrained (true) score network f_phi

    # *********
    true_score = dnnlib.util.construct_class_by_name(**network_kwargs,
                                                     **interface_kwargs)  # subclass of torch.nn.Module
    # true_score = dnnlib.util.construct_class_by_name(**generator_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    # *********

    true_score.eval().requires_grad_(False).to(device)

    # Construct the generator (fake) score network f_psi
    fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device)

    # Construct the generator G_theta

    G = dnnlib.util.construct_class_by_name(**generator_kwargs,
                                            **interface_kwargs)  # subclass of torch.nn.Module
    # G = dnnlib.util.construct_class_by_name(**generator_kwargs, **interface_kwargs_generator) # subclass of torch.nn.Module

    G.train().requires_grad_(True).to(device)
    # G = copy.deepcopy(true_score).train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, true_score.img_channels, true_score.img_resolution,
                                  true_score.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, true_score.label_dim], device=device)
            misc.print_module_summary(true_score, [images, sigma, labels], max_nesting=2)

    # Setup loss function and augment_pipe
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    augment_pipe = dnnlib.util.construct_class_by_name(
        **augment_kwargs) if augment_kwargs is not None else None  # training.augment.AugmentPipe

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from URL "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow

        dist.print0('Loading network completed')
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=true_score,
                                     require_all=False)

        if resume_training is not None:
            del data
            dist.print0('checkpoint path:', resume_training)

            if dist.get_rank() != 0:
                torch.distributed.barrier()  # rank 0 goes first

            checkpoint = torch.load(resume_training, map_location=torch.device('cpu'))

            if dist.get_rank() == 0:
                torch.distributed.barrier()  # other ranks follow

            dist.print0('Setting up optimizer...')
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score,
                                                                       device_ids=[device])
            fake_score_ddp.module.load_state_dict(checkpoint['fake_score_state'])
            fake_score_optimizer = dnnlib.util.construct_class_by_name(
                params=fake_score_ddp.module.parameters(), **fake_score_optimizer_kwargs)
            fake_score_optimizer.load_state_dict(checkpoint['fake_score_optimizer_state'])

            G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device])
            G_ddp.module.load_state_dict(checkpoint['G_state'])
            g_optimizer = dnnlib.util.construct_class_by_name(params=G_ddp.module.parameters(),
                                                              **g_optimizer_kwargs)
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])

            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            G_ema.load_state_dict(checkpoint['G_ema_state'])
            G_ema.eval().requires_grad_(False)

            torch.distributed.barrier()
            del checkpoint  # conserve memory
            dist.print0('Loading checkpoint completed')
        else:
            # Setup optimizer.
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=fake_score,
                                         require_all=False)
            if sid_model is not None:
                if dist.get_rank() != 0:
                    torch.distributed.barrier()  # rank 0 goes first
                with dnnlib.util.open_url(sid_model, verbose=(dist.get_rank() == 0)) as f:
                    data = pickle.load(f)
                if dist.get_rank() == 0:
                    torch.distributed.barrier()
            # misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G, require_all=True)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G, require_all=False)

            dist.print0('Setting up optimizer...')
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score,
                                                                       device_ids=[device])
            fake_score_optimizer = dnnlib.util.construct_class_by_name(
                params=fake_score_ddp.module.parameters(), **fake_score_optimizer_kwargs)

            G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device])
            g_optimizer = dnnlib.util.construct_class_by_name(params=G_ddp.module.parameters(),
                                                              **g_optimizer_kwargs)

            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G_ema,
                                         require_all=False)

            torch.distributed.barrier()
            del data  # conserve memory

        fake_score_ddp.eval().requires_grad_(False)
        G_ddp.eval().requires_grad_(False)

    # dist.print0('Exporting sample images...')
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None

    stats_metrics = dict()

    if dist.get_rank() == 0:
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj,
                                                              training_loader=dataloader_for_grid,
                                                              label_dim=train_dataset_ref.label_dim,
                                                              has_labels=train_dataset_ref.has_labels,
                                                              )

        _, grid_indices = setup_snapshot_image_grid_decode(training_set=dataset_obj,
                                                           training_loader=dataloader_for_grid,
                                                           label_dim=train_dataset_ref.label_dim,
                                                           has_labels=train_dataset_ref.has_labels,
                                                           encoder=None,
                                                           device=device)

        grid_z = init_sigma * torch.randn(
            [labels.shape[0], G_ema.img_channels, G_ema.img_resolution, G_ema.img_resolution],
            device=device)
        # grid_z = grid_z.split(batch_gpu)
        grid_z = grid_z.split(1)

        grid_c = torch.from_numpy(labels).to(device)
        # grid_c = grid_c.split(batch_gpu)
        grid_c = grid_c.split(1)

        if resume_training is None:
            with torch.no_grad():
                print('Exporting sample images...')
                # # images = torch.cat([encoder.decode(G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                # if use_gan:
                #     images = [vae_edm2_decode(vae, scale, bias, vae_edm2_encode_latents(scale, bias,
                #                                                                         torch.as_tensor(
                #                                                                             input).to(
                #                                                                             device).unsqueeze(
                #                                                                             0))).to(
                #         vaedevice) for input, _ in (dataset_obj[i] for i in grid_indices)]
                #     images = torch.cat(images).cpu().numpy()
                #     # images = torch.cat([encoder.decode(encoder.vae_edm2_encode_latents(torch.as_tensor(input).to(device).unsqueeze(0))).cpu() for input, _ in (dataset_obj[i] for i in grid_indices)]).numpy()
                #     # labels = [label for _, label in (training_set[i] for i in grid_indices)]
                #     save_image_grid(img=images, fname=os.path.join(run_dir, 'reals.png'),
                #                     drange=[0, 255], grid_size=grid_size)
                #
                # # images = torch.cat([encoder.decode(encoder.encode_random_latents(G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c))).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                # # images = torch.cat([encoder.decode_sid(G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                #
                # images = [vae_edm2_decode(vae, scale, bias, G_ema(z, (
                #             init_sigma * torch.ones(z.shape[0], 1, 1, 1)).to(z.device), c).to(
                #     vaedevice)) for z, c in zip(grid_z, grid_c)]
                # images = torch.cat(images).cpu().numpy()
                # # save_image_grid(img=images, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
                #
                # save_image_grid(img=images, fname=os.path.join(run_dir, 'fakes_init.png'),
                #                 drange=[0, 255], grid_size=grid_size)
                # del images

    # Set random seed
    #     if resume_training is not None:
    #         seed = (seed+resume_kimg)% (1 << 31)
    # np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    # torch.manual_seed(np.random.randint(1 << 31))

    if metrics is not None:
        dist.print0('Evaluating metrics...')
        for metric in metrics:

            result_dict = calculate_metric(
                metric=metric,
                G=G_ema,
                vae=vae,
                scale=scale,
                bias=bias,
                dtype=torch.float16,
                train_sampler=False,
                num_steps=1,
                # G=partial(vae_sampler_edm2, vae=vae, scale=scale, bias=bias, unet=G_ema, dtype=torch.float16, train_sampler=False, num_steps=1),
                init_sigma=init_sigma,
                dataset_kwargs=dataset_kwargs,
                num_gpus=dist.get_world_size(),
                rank=dist.get_rank(),
                local_rank=dist.get_local_rank(),
                device=device,
                data_stat=data_stat,
                detector_url=detector_url,
                batch_size=64,
                batch_gen=4,
            )

            if dist.get_rank() == 0:
                print(result_dict.results)
                metric_main.report_metric(result_dict, run_dir=run_dir,
                                          snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg // 1000:06d}.png',
                                          alpha=alpha)
                if metric == 'fid50k_full':
                    current_fid = result_dict.results.fid50k_full
            stats_metrics.update(result_dict.results)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()

    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    # stats_metrics = dict()

    data = dict(ema=G_ema)
    for key, value in data.items():
        if isinstance(value, torch.nn.Module):
            value = copy.deepcopy(value).eval().requires_grad_(False)
            # misc.check_ddp_consistency(value)
            data[key] = value.cpu()
        del value  # conserve memory

    if dist.get_rank() == 0:
        save_data(data=data, fname=os.path.join(run_dir,
                                                f'network-snapshot-{alpha:03f}-{cur_nimg // 1000:06d}.pkl'))

    del data  # conserve memory
    dist.print0('Exporting sample images...')

    if save_best_and_last:
        if dist.get_rank() == 0:
            best_fid = float('inf')
            current_fid = float('inf')
            previous_best_pkl_filename = None
            previous_pt_filename = None

    while True:
        # Update fake score network f_psi
        # Accumulate gradients.
        fake_score_ddp.train().requires_grad_(True)
        fake_score_optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            real_images, labels = next(dataset_iterator)
            if use_gan:
                with torch.no_grad():
                    real_images = vae_edm2_encode_latents(scale, bias, real_images.to(device))
                z = init_sigma * torch.randn_like(real_images)
            else:
                z = init_sigma * torch.randn(len(labels), 4, 64, 64).to(device)

            # real_images = encoder.vae_edm2_encode_latents(real_images.to(device))

            # real_images = real_images.to(device).to(torch.float32) / 127.5 - 1

            labels = labels.to(device)

            # torch.autograd.set_detect_anomaly(True)
            with misc.ddp_sync(G_ddp, False):
                with torch.no_grad():
                    images = G_ddp(z, init_sigma * torch.ones(z.shape[0], 1, 1, 1).to(z.device),
                                   labels)
                    # images = encoder.encode_random_latents(images)
                with misc.ddp_sync(fake_score_ddp, (round_idx == num_accumulation_rounds - 1)):

                    if use_gan == False:
                        # dist.print0('no issue')
                        loss = loss_fn(fake_score=fake_score_ddp, images=images, labels=labels,
                                       return_logvar=return_logvar)
                        loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                        loss.backward()
                        lossD_print = 0
                    else:
                        # dist.print0('warmning!')
                        loss, loss_D = loss_fn.fakescore_discriminator_share_encoder_loss(
                            fake_score=fake_score_ddp, images=images, labels=labels,
                            real_images=real_images, true_score=true_score, alpha=alpha,
                            return_logvar=return_logvar)
                        loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                        loss_D = loss_D.sum().mul(loss_scaling_D / batch_gpu_total)
                        lossAll = loss + loss_D
                        lossAll.backward()
                        lossD_print = loss_D.item()

        loss_fake_score_print = loss.item()
        training_stats.report('fake_score_Loss/loss', loss_fake_score_print)
        training_stats.report('D_Loss/loss', lossD_print)

        fake_score_ddp.eval().requires_grad_(False)

        for param in fake_score_ddp.module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                # torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

        # torch.nn.utils.clip_grad_value_(fake_score_ddp.module.parameters(), 1)

        fake_score_optimizer.step()
        fake_score_optimizer.zero_grad(set_to_none=True)

        # Update generator G_theta
        G_ddp.train().requires_grad_(True)
        g_optimizer.zero_grad(set_to_none=True)
        with misc.ddp_sync(fake_score_ddp, False):
            for round_idx in range(num_accumulation_rounds):
                images, labels = next(dataset_iterator)
                if use_gan:
                    images = vae_edm2_encode_latents(scale, bias, images.to(device))
                    z = init_sigma * torch.randn_like(images)
                else:
                    z = init_sigma * torch.randn(len(labels), 4, 64, 64).to(device)
                # images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                with misc.ddp_sync(G_ddp, (round_idx == num_accumulation_rounds - 1)):
                    images = G_ddp(z, init_sigma * torch.ones(z.shape[0], 1, 1, 1).to(z.device),
                                   labels)
                    # images = encoder.encode_random_latents(images)
                    # images = encoder.encode_random_latents(images)

                    if use_gan == False:
                        loss = loss_fn.generator_loss(true_score=true_score,
                                                      fake_score=fake_score_ddp, images=images,
                                                      labels=labels, alpha=alpha, tmax=tmax,
                                                      return_logvar=return_logvar)
                        lossG_gan_print = 0
                    else:
                        loss, loss_gan = loss_fn.generator_share_encoder_loss(true_score=true_score,
                                                                              fake_score=fake_score_ddp,
                                                                              images=images,
                                                                              labels=labels,
                                                                              alpha=alpha,
                                                                              tmax=tmax,
                                                                              return_logvar=return_logvar)
                        loss_gan = loss_gan.mean(dim=[1, 2, 3], keepdim=True) * images.shape[1] * \
                                   images.shape[2] * images.shape[3]
                        loss_gan = loss_gan.sum().mul(loss_scaling_G_gan / batch_gpu_total)
                        lossG_gan_print = loss_gan.item()

                    loss = loss.sum().mul(loss_scaling_G / batch_gpu_total)

                    if use_gan == True and cur_nimg > 200 * 1000:
                        (0.5 * loss + 0.5 * loss_gan).backward()
                    else:
                        loss.backward()
        lossG_print = loss.item()
        training_stats.report('G_Loss/loss', lossG_print)

        training_stats.report('G_gan_Loss/loss', lossG_gan_print)

        G_ddp.eval().requires_grad_(False)

        for param in G_ddp.module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                # torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        if use_gan == False or cur_nimg > 100 * 1000:
            g_optimizer.step()
        g_optimizer.zero_grad(set_to_none=True)

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))

        for p_ema, p_true_score in zip(G_ema.parameters(), G.parameters()):
            # p_ema.copy_(p_true_score.detach().lerp(p_ema, ema_beta))
            p_ema.lerp_(p_true_score.detach(), 1 - ema_beta)

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"loss_fake_score {training_stats.report0('fake_score_Loss/loss', loss_fake_score_print):<6.2f}"]
        fields += [f"loss_D {training_stats.report0('D_Loss/loss', lossD_print):<6.2f}"]
        fields += [f"loss_G {training_stats.report0('G_Loss/loss', lossG_print):<6.2f}"]
        fields += [f"loss_G_gan {training_stats.report0('G_gan_Loss/loss', lossG_gan_print):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0 or
                                             cur_tick in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):

            dist.print0('Exporting sample images...')
            # if dist.get_rank() == 0:
            #     # images = torch.cat([encoder.decode(G_ema(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device), c)).to(z.device).to(z.dtype).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            #
            #     # images = torch.cat([vae_edm2_decode(vae,scale,bias,G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            #
            #     images = [vae_edm2_decode(vae, scale, bias, G_ema(z, (
            #             init_sigma * torch.ones(z.shape[0], 1, 1, 1)).to(z.device), c).to(
            #         vaedevice)) for z, c in zip(grid_z, grid_c)]
            #     images = torch.cat(images).cpu().numpy()
            #
            #     # images = torch.cat([encoder.decode(encoder.encode_random_latents(G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c))).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            #     # images = torch.cat([encoder.decode_sid(G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            #
            #     # save_image_grid(img=images, fname=os.path.join(run_dir, f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            #     save_image_grid(img=images, fname=os.path.join(run_dir,
            #                                                    f'fakes_{alpha:03f}_{cur_nimg // 1000:06d}.png'),
            #                     drange=[0, 255], grid_size=grid_size)
            #     del images

            if metrics is not None and cur_tick >= 100:
                dist.print0('Evaluating metrics...')
                for metric in metrics:

                    result_dict = calculate_metric(
                        metric=metric,
                        G=G_ema,
                        vae=vae,
                        scale=scale,
                        bias=bias,
                        dtype=torch.float16,
                        train_sampler=False,
                        num_steps=1,
                        # G=partial(vae_sampler_edm2, vae=vae, scale=scale, bias=bias, unet=G_ema, dtype=torch.float16, train_sampler=False, num_steps=1),
                        init_sigma=init_sigma,
                        dataset_kwargs=dataset_kwargs,
                        num_gpus=dist.get_world_size(),
                        rank=dist.get_rank(),
                        local_rank=dist.get_local_rank(),
                        device=device,
                        data_stat=data_stat,
                        detector_url=detector_url,
                        batch_size=64,
                        batch_gen=1,
                    )

                    if dist.get_rank() == 0:
                        print(result_dict.results)
                        metric_main.report_metric(result_dict, run_dir=run_dir,
                                                  snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg // 1000:06d}.png',
                                                  alpha=alpha)
                        if metric == 'fid50k_full':
                            current_fid = result_dict.results.fid50k_full
                    stats_metrics.update(result_dict.results)

            data = dict(ema=G_ema)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    data[key] = value.cpu()
                del value  # conserve memory

            if dist.get_rank() == 0:  # Check if this is the master process
                if save_best_and_last is False:
                    save_data(data=data, fname=os.path.join(run_dir,
                                                            f'network-snapshot-{alpha:03f}-{cur_nimg // 1000:06d}.pkl'))
                else:
                    if current_fid < best_fid:  # Check if the current model is better
                        best_fid = current_fid
                        new_best_pkl_filename = os.path.join(run_dir,
                                                             f'network-snapshot-{alpha:03f}-{cur_nimg // 1000:06d}.pkl')
                        save_data(data=data, fname=new_best_pkl_filename)  # Save the new best model
                        try:
                            if previous_best_pkl_filename is not None:
                                if os.path.exists(previous_best_pkl_filename):
                                    os.remove(
                                        previous_best_pkl_filename)  # Remove the previous best model file
                        except OSError as e:
                            dist.print0(f"Error removing previous pkl: {e}")
                        previous_best_pkl_filename = new_best_pkl_filename  # Update the reference to the best model file                     

            del data  # conserve memory

        if (state_dump_ticks is not None) and (
                done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            dist.print0(f'saving checkpoint: training-state-{cur_nimg // 1000:06d}.pt')

            # save_pt(pt=dict(fake_score=fake_score, G=G, G_ema=G_ema, fake_score_optimizer_state=fake_score_optimizer.state_dict(), g_optimizer_state=g_optimizer.state_dict()), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

            save_pt(pt=dict(fake_score_state=fake_score_ddp.module.state_dict(),
                            G_state=G_ddp.module.state_dict(), G_ema_state=G_ema.state_dict(),
                            fake_score_optimizer_state=fake_score_optimizer.state_dict(),
                            g_optimizer_state=g_optimizer.state_dict()),
                    fname=os.path.join(run_dir, f'training-state-{cur_nimg // 1000:06d}.pt'))

            if save_best_and_last is True:
                try:
                    if previous_pt_filename is not None:
                        if os.path.exists(previous_pt_filename):
                            os.remove(previous_pt_filename)
                except OSError as e:
                    dist.print0(f"Error removing previous checkpoint: {e}")

                previous_pt_filename = os.path.join(run_dir,
                                                    f'training-state-{cur_nimg // 1000:06d}.pt')

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                append_line(jsonl_line=json.dumps(
                    dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n',
                            fname=os.path.join(run_dir, f'stats_{alpha:03f}.jsonl'))

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

    return G_ema, None

# ----------------------------------------------------------------------------
