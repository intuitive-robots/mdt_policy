from typing import Callable, List
import math 

import numpy as np
import torch
import torch.nn as nn
import einops 
from voltron import instantiate_extractor, load



def ensure_list(value):
    if isinstance(value, str):  # Check if value is a string
        return [value]  # Return the string as a single element in a list
    elif isinstance(value, list):  # Check if value is already a list
        return value  # Return the list as it is
    else:
        raise ValueError("Input must be a string or a list.")  # Raise an error for other types


class VoltronEncoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 512,
        model_type: str = 'v-cond',
        device: str = 'cuda:0',
        cache: str = './pretrained_model_weights'
    ):
        super(VoltronEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.model_type = model_type
        self.vcond, self.preprocess = load(model_type, device=device, freeze=True, cache=cache)
        if self.model_type == 'v-dual':
            self.vector_extractor = instantiate_extractor(self.vcond, n_latents=2, output_dim=latent_dim, n_heads=4)().to(device)
        else:
            self.vector_extractor = instantiate_extractor(self.vcond, n_latents=1, output_dim=latent_dim, n_heads=4)().to(device)
        for param in self.vcond.parameters():
            param.requires_grad = False
    
    def forward(self, x, lang=None):
        batch_size = len(x)
        t_steps = 1
        time_series = False
        if len(x.shape) == 5:
            t_steps = x.shape[1]
            x = einops.rearrange(x, 'b t n x_dim y_dim -> (b t) n x_dim y_dim')
            # print(f'After rearrange x shape: {x.shape}')
            time_series = True
            
        x = self.preprocess(x)
        if self.model_type == 'v-dual':
            x = einops.rearrange(x, '(b t) n x_dim y_dim -> b t n x_dim y_dim', t=2)
                
        with torch.no_grad():
            if lang is not None:
                lang = ensure_list(lang)
                x = self.vcond(x, lang, mode="multimodal")
            else:
                x = self.vcond(x)
        x = self.vector_extractor(x)
        if time_series:
            if self.model_type == 'v-dual':
                pass 
                # x = einops.rearrange(x, '(b 1) d -> b 1 d')
            else:
                x = einops.rearrange(x, '(b t) d -> b t d', b=batch_size, t=t_steps, d=self.latent_dim)
        # print(x.shape)
        return x


class VoltronTokenEncoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 512,
        model_type: str = 'v-cond',
        device: str = 'cuda:0',
        cache: str = './pretrained_model_weights'
    ):
        super(VoltronTokenEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.vcond, self.preprocess = load(model_type, device=device, freeze=True, cache=cache)
        self.device = device
        
        self.vcond.eval()
        for param in self.vcond.parameters():
            param.requires_grad = False
    
    def forward(self, x, lang=None):
        batch_size = len(x)
        t_steps = 1
        time_series = False
        if len(x.shape) == 5:
            t_steps = x.shape[1]
            x = einops.rearrange(x, 'b t n x_dim y_dim -> (b t) n x_dim y_dim')
            time_series = True
            
        x = self.preprocess(x)
        with torch.no_grad():
            if lang is not None:
                lang = ensure_list(lang)
                x = self.vcond(x.to(next(self.vcond.parameters()).device), lang, mode="multimodal")
            else:
                x = self.vcond(x, mode='visual')
        return x