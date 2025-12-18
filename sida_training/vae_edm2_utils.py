# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


import torch
import diffusers

from diffusers import AutoencoderKL

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    XFormersAttnProcessor,
    LoRAXFormersAttnProcessor,
    LoRAAttnProcessor2_0,
    FusedAttnProcessor2_0,
)


def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
            FusedAttnProcessor2_0,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)


def load_sd_vae(pretrained_vae_model_name_or_path, device):  # , weight_dtype):
    # Load the tokenizer
    print(f'pretrained_model_name_or_path: {pretrained_vae_model_name_or_path}')

    # print('tokenizer start')

    vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path)

    # Freeze untrained components
    vae.eval().requires_grad_(False).to(device)

    return vae


# def encode_latents(vae, x): # raw latents => final latents
#     raw_mean    = [5.81, 3.25, 0.12, -2.15],    # Assumed mean of the raw latents.
#     raw_std     = [4.17, 4.62, 3.71, 3.28],     # Assumed standard deviation of the raw latents.
#     final_mean  = 0,                            # Desired mean of the final latents.
#     final_std   = 0.5,                          # Desired standard deviation of the final latents.
#         scale = np.float32(final_std) / np.float32(raw_std)
#         self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
#         self.batch_size = int(batch_size)
#         self._vae = None


#     mean, std = x.to(torch.float32).chunk(2, dim=1)
#     x = mean + torch.randn_like(mean) * std
#     x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
#     x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
#     return x


# def vae_edm2_decode(vae, latents):

#     # make sure the VAE is in float32 mode, as it overflows in float16
#     needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
#     if needs_upcasting:
#         upcast_vae(vae=vae)
#         latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
#     images = vae.decode(latents, return_dict=False)[0]
#     #images = vae.decode(D_x /0.18215).sample
#     # cast back to fp16 if needed
#     if needs_upcasting:
#         vae.to(dtype=torch.float16)
#     return images.to(torch.float32)


# def vae_sampler_edm2(vae,scale,bias, G,z,init_sigma,c)
#     latent_image = G(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c)
#     images=vae_edm2_decode(vae,scale,bias,latent_image)
#     return images.to(torch.float32)


# def vae_sampler_edm2(vae,scale,bias,unet,latents, c,init_sigma=2.5, dtype=torch.float16,train_sampler=True,num_steps=1):

#     z=latents
#     G=unet
#     if train_sampler:
#         D_x = G(z,(init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c) #.to(torch.float32)
#     else:
#         with torch.no_grad():
#             D_x = G(z,(init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c) #.to(torch.float32)
#     needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
#     if needs_upcasting:
#         upcast_vae(vae=vae)
#         D_x = D_x.to(next(iter(vae.post_quant_conv.parameters())).dtype)

#     images=vae_edm2_decode(vae,scale,bias,D_x).to(torch.float32)
#     return images


# def vae_sampler_edm2(vae, scale, bias, unet, latents, c, init_sigma=2.5, dtype=torch.float16, train_sampler=True, num_steps=1):
#     z = latents
#     G = unet

#     # Initialize sigma tensor
#     init_sigma_tensor = init_sigma * torch.ones(z.shape[0], 1, 1, 1, device=z.device)

#     # Forward pass
#     if train_sampler:
#         D_x = G(z, init_sigma_tensor, c)
#     else:
#         with torch.no_grad():
#             D_x = G(z, init_sigma_tensor, c)

#     # Check if VAE needs upcasting
#     needs_upcasting = (getattr(vae, 'dtype', None) == torch.float16) and getattr(vae.config, 'force_upcast', False)
#     if needs_upcasting:
#         upcast_vae(vae=vae)
#         D_x = D_x.to(next(iter(vae.post_quant_conv.parameters())).dtype)

#     # Decode images
#     images = vae_edm2_decode(vae, scale, bias, D_x).to(torch.float32)

#     return images


def vae_sampler_edm2(vae, scale, bias, unet, latents, c, init_sigma_tensor, dtype=torch.float16,
                     train_sampler=True, num_steps=1):
    z = latents
    G = unet

    # Forward pass
    if train_sampler:
        D_x = G(z, init_sigma_tensor, c)
    else:
        with torch.no_grad():
            D_x = G(z, init_sigma_tensor, c)

    # Check if VAE needs upcasting
    needs_upcasting = (getattr(vae, 'dtype', None) == torch.float16) and getattr(vae.config,
                                                                                 'force_upcast',
                                                                                 False)
    if needs_upcasting:
        upcast_vae(vae=vae)
        D_x = D_x.to(next(iter(vae.post_quant_conv.parameters())).dtype)

    # Decode images
    images = vae_edm2_decode(vae, scale, bias, D_x).to(torch.float32)

    return images


