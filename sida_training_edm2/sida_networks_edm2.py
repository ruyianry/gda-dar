# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
#from edm2.networks_edm2 import resample, mp_silu, mp_sum, mp_cat, MPFourier, Block,MPConv, normalize

if 1:
    from edm2.networks_edm2 import resample, mp_silu, mp_sum, mp_cat, MPFourier,Block #,normalize,Block #, Block # MPConv, normalize, 


    # def normalize(x, dim=None, eps=1e-4):
    #     if dim is None:
    #         dim = list(range(1, x.ndim))
    #     norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    #     norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    #     return (x / norm).to(x.dtype)
    #     #return x


    def normalize(x, dim=None, eps=1e-4):
        if dim is None:
            dim = list(range(1, x.ndim))
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
        norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
        #norm = torch.clamp(norm, min=eps)  # Add clamping to prevent division by zero
        if torch.isnan(norm).any() or torch.isinf(norm).any():
            print("Warning: NaN or Inf detected in norm calculation!")
        return x / norm.to(x.dtype)



#     @persistence.persistent_class
#     class Block(torch.nn.Module):
#         def __init__(self,
#             in_channels,                    # Number of input channels.
#             out_channels,                   # Number of output channels.
#             emb_channels,                   # Number of embedding channels.
#             flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
#             resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
#             resample_filter     = [1,1],    # Resampling filter.
#             attention           = False,    # Include self-attention?
#             channels_per_head   = 64,       # Number of channels per attention head.
#             dropout             = 0,        # Dropout probability.
#             res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
#             attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
#             clip_act            = 256,      # Clip output activations. None = do not clip.
#         ):
#             super().__init__()
#             self.out_channels = out_channels
#             self.flavor = flavor
#             self.resample_filter = resample_filter
#             self.resample_mode = resample_mode
#             self.num_heads = out_channels // channels_per_head if attention else 0
#             self.dropout = dropout
#             self.res_balance = res_balance
#             self.attn_balance = attn_balance
#             self.clip_act = clip_act
#             self.emb_gain = torch.nn.Parameter(torch.zeros([]))
#             self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
#             self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
#             self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
#             self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
#             self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
#             self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

#         def forward(self, x, emb):
#             # Main branch.
#             x = resample(x, f=self.resample_filter, mode=self.resample_mode)
#             if self.flavor == 'enc':
#                 if self.conv_skip is not None:
#                     x = self.conv_skip(x)
#                 x = normalize(x, dim=1) # pixel norm

#             # Residual branch.
#             y = self.conv_res0(mp_silu(x))
#             c = self.emb_linear(emb, gain=self.emb_gain) + 1
#             y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
#             if self.training and self.dropout != 0:
#                 y = torch.nn.functional.dropout(y, p=self.dropout)
#             y = self.conv_res1(y)

#             # Connect the branches.
#             if self.flavor == 'dec' and self.conv_skip is not None:
#                 x = self.conv_skip(x)
#             x = mp_sum(x, y, t=self.res_balance)

#             # Self-attention.
#             # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
#             # but we haven't done sufficient testing to verify that it produces identical results.
#             if self.num_heads != 0:
#                 y = self.attn_qkv(x)
#                 y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
#                 q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
#                 w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
#                 y = torch.einsum('nhqk,nhck->nhcq', w, v)
#                 y = self.attn_proj(y.reshape(*x.shape))
#                 x = mp_sum(x, y, t=self.attn_balance)

#             # Clip activations.
#             if self.clip_act is not None:
#                 x = x.clip_(-self.clip_act, self.clip_act)
#             return x


    # @persistence.persistent_class
    # class MPConv(torch.nn.Module):
    #     def __init__(self, in_channels, out_channels, kernel):
    #         super().__init__()
    #         self.out_channels = out_channels
    #         self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    #     def forward(self, x, gain=1):
    #         w = self.weight.to(torch.float32)

    #         if self.training:
    #             with torch.no_grad():
    #                 self.weight.copy_(normalize(w)) # forced weight normalization


    #         w = normalize(w) # traditional weight normalization
    #         w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
    #         w = w.to(x.dtype)
    #         if w.ndim == 2:
    #             return x @ w.t()
    #         assert w.ndim == 4
    #         return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


    @persistence.persistent_class
    class MPConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel):
            super().__init__()
            self.out_channels = out_channels
            self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

            # Register the forward pre-hook
            
            #self.register_forward_pre_hook(self._apply_forced_weight_normalization)

        # def _apply_forced_weight_normalization(self, module, input):
        #     # Only apply during training
        #     if self.training:
        #         with torch.no_grad():
        #             w = self.weight.to(torch.float32)
        #             w_normalized = normalize(w)
        #             self.weight.copy_(w_normalized)

        def forward(self, x, gain=1):
            w = self.weight.to(torch.float32)
            w = normalize(w)  # Traditional weight normalization
            w = w * (gain / np.sqrt(w[0].numel()))  # Magnitude-preserving scaling
            w = w.to(x.dtype)
            if w.ndim == 2:
                return x @ w.t()
            assert w.ndim == 4
            return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2))    
    
    
@persistence.persistent_class
class UNet_EncoderDecoder(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

    def forward(self, x, noise_labels, class_labels,return_flag='decoder'):
        assert return_flag in ['encoder', 'decoder', 'encoder_decoder'], f"Invalid return_flag: {return_flag}"
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            if return_flag !='encoder':
                skips.append(x)
        #print(x)    
        if return_flag in ['encoder','encoder_decoder']:
            logits= x.mean(dim=1, keepdim=True)
            #print(logits)
        if return_flag=='encoder':
            return logits
        
        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        
        if return_flag=='decoder':
            return x
        else:
            return x, logits
    
    

"""Model architectures and preconditioning schemes used in the paper
"Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step"."""

    
@persistence.persistent_class
class EDM_2_Precond_EncoderDecoder(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        ##
        self.unet = UNet_EncoderDecoder(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs)
        ##
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, return_logvar=False, return_flag = 'decoder', **unet_kwargs):
        assert return_flag in ['decoder', 'encoder', 'encoder_decoder','generator'], f"Invalid return_flag: {return_flag}"
        
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
    
        x_in = (c_in * x).to(dtype)
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
        #logvar = 0
        if return_flag in ['decoder', 'encoder']:
            
            F_x = self.unet(x_in, c_noise, class_labels, return_flag =return_flag, **unet_kwargs)
            
            if return_flag=='decoder':
                D_x = c_skip * x + c_out * F_x.to(torch.float32)
                if return_logvar:
                    return D_x, logvar # u(sigma) in Equation 21
                return D_x
            else:
                #encoder
                logits =  F_x.to(torch.float32)
                if return_logvar:
                    return logits,logvar
                return logits
        else:
            #'encoder_decoder'
            F_x,logits = self.unet(x_in, c_noise, class_labels, return_flag =return_flag, **unet_kwargs)
            logits = logits.to(torch.float32)
            D_x = c_skip * x + c_out * F_x.to(torch.float32)
            if return_logvar:
                return D_x,logits,logvar
            return D_x,logits

        
@persistence.persistent_class
class EDM_2_Precond_Generator(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        ##
        self.unet = UNet_EncoderDecoder(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs)
        ##
        #self.logvar_fourier = MPFourier(logvar_channels)
        #self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, class_labels, force_fp32=False, return_flag = 'decoder', augment_labels=None,**unet_kwargs):
        assert return_flag in ['decoder', 'encoder', 'encoder_decoder','generator'], f"Invalid return_flag: {return_flag}"
        
        # x = x.to(torch.float32)
        # sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        # dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # # Preconditioning weights.
        # c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        # c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        # c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        # c_noise = sigma.flatten().log() / 4

        # # Run the model.
    
        # x_in = (c_in * x).to(dtype)
        # #if return_logvar:
        # #    logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
        # #logvar = 0
        # F_x = self.unet(x_in, c_noise, class_labels, return_flag =return_flag, **unet_kwargs)
        # D_x = c_skip * x + c_out * F_x.to(torch.float32)
        # #D_x = F_x.to(torch.float32)
        # return D_x 

        #assert return_flag in ['decoder', 'encoder', 'encoder_decoder','generator'], f"Invalid return_flag: {return_flag}"
        
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
    
        x_in = (c_in * x).to(dtype)
        # if return_logvar:
        #     logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
        # #logvar = 0
        if return_flag in ['decoder', 'encoder']:
            
            F_x = self.unet(x_in, c_noise, class_labels, return_flag =return_flag, **unet_kwargs)
            
            if return_flag=='decoder':
                D_x = c_skip * x + c_out * F_x.to(torch.float32)
                # if return_logvar:
                #     return D_x, logvar # u(sigma) in Equation 21
                return D_x
            else:
                #encoder
                logits =  F_x.to(torch.float32)
                # if return_logvar:
                #     return logits,logvar
                return logits
        else:
            #'encoder_decoder'
            F_x,logits = self.unet(x_in, c_noise, class_labels, return_flag =return_flag, **unet_kwargs)
            logits = logits.to(torch.float32)
            D_x = c_skip * x + c_out * F_x.to(torch.float32)
            # if return_logvar:
            #     return D_x,logits,logvar
            return D_x,logits