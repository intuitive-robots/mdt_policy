import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops
from einops import rearrange, repeat, reduce

from mdt.models.networks.transformers.transformer_blocks import *


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


logger = logging.getLogger(__name__)

def return_model_parameters_in_millions(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_in_millions = round(num_params / 1_000_000, 2)
    return num_params_in_millions


class MDTTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        n_dec_layers: int,
        n_enc_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        proprio_dim: Optional[int] = None,
        goal_drop: float = 0.1,
        bias=False,
        use_abs_pos_emb: bool = True,
        use_rot_embed: bool = False,
        rotary_xpos: bool = False,
        linear_output: bool = True,
        use_ada_conditioning: bool = False,
        use_noise_encoder: bool = False,
        latent_is_decoder: bool = False,
        use_modality_encoder: bool = False,
        use_mlp_goal: bool = False,
    ):
        super().__init__()
        self.device = device
        self.goal_conditioned = goal_conditioned
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.use_ada_conditioning = use_ada_conditioning
        self.proprio_dim = proprio_dim
        self.latent_is_decoder = latent_is_decoder
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        self.action_seq_len = action_seq_len
        self.use_modality_encoder = use_modality_encoder
        seq_size = goal_seq_len + action_seq_len
        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.incam_embed = nn.Linear(self.obs_dim, self.embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.cond_mask_prob = goal_drop
        self.use_rot_embed = use_rot_embed
        self.use_abs_pos_emb = use_abs_pos_emb
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.latent_encoder_emb = None

        if use_mlp_goal:
            self.goal_emb = nn.Sequential(
                nn.Linear(goal_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.goal_emb = nn.Linear(goal_dim, embed_dim)
        if self.use_modality_encoder:
            if use_mlp_goal:
                self.lang_emb = nn.Sequential(
                    nn.Linear(goal_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            else:
                self.lang_emb = nn.Linear(goal_dim, embed_dim)
        else:
            self.lang_emb = self.goal_emb

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_enc_layers,
            block_size=block_size,
            bias=bias,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            mlp_pdrop=mlp_pdrop,
        )

        if self.use_ada_conditioning:
            self.decoder = TransformerFiLMDecoder(
                embed_dim=embed_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                n_layers=n_dec_layers,
                film_cond_dim=embed_dim,
                block_size=block_size,
                bias=bias,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                mlp_pdrop=mlp_pdrop,
                use_cross_attention=True,
                use_noise_encoder=use_noise_encoder,
            )
        else:
            self.decoder = TransformerDecoder(
                embed_dim=embed_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                n_layers=n_dec_layers,
                block_size=block_size,
                bias=bias,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                mlp_pdrop=mlp_pdrop,
                use_cross_attention=True,
            )

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len

        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        ).to(self.device)

        self.action_emb = nn.Linear(action_dim, embed_dim)

        if linear_output:
            self.action_pred = nn.Linear(embed_dim, self.action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, self.action_dim)
            )

        if proprio_dim is not None:
            self.proprio_emb = nn.Sequential(
                nn.Linear(proprio_dim, embed_dim * 2),
                nn.Mish(),
                nn.Linear(embed_dim * 2, embed_dim),
            ).to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, MDTTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states, actions, goals, sigma, uncond: Optional[bool] = False):
        context = self.enc_only_forward(states, actions, goals, sigma, uncond)
        pred_actions = self.dec_only_forward(context, actions, sigma)
        return pred_actions

    def enc_only_forward(self, states, actions, goals, sigma, uncond: Optional[bool] = False):
        emb_t = self.process_sigma_embeddings(sigma) if not self.use_ada_conditioning else None
        goals = self.preprocess_goals(goals, 1, uncond)
        state_embed, proprio_states = self.process_state_embeddings(states)
        goal_embed = self.goal_emb(goals)
        action_embed = self.action_emb(actions)

        if self.use_abs_pos_emb:
            goal_x, state_x, action_x, proprio_x = self.apply_position_embeddings(goal_embed, state_embed, action_embed, proprio_states, 1)
        else:
            goal_x = self.drop(goal_embed)
            state_x = self.drop(state_embed)
            action_x = self.drop(action_embed)
            proprio_x = self.drop(proprio_states) if proprio_states is not None else None

        input_seq = self.concatenate_inputs(emb_t, goal_x, state_x, action_x, proprio_x, uncond)
        context = self.encoder(input_seq)
        self.latent_encoder_emb = context
        return context

    def dec_only_forward(self, context, actions, sigma):
        emb_t = self.process_sigma_embeddings(sigma)
        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed)

        if self.use_ada_conditioning:
            x = self.decoder(action_x, emb_t, context)
        else:
            x = self.decoder(action_x, context)

        pred_actions = self.action_pred(x)
        return pred_actions
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()

    def forward_enc_only(self, states, actions, goals, sigma, uncond: Optional[bool] = False):
        b, t, dim = states['static'].size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        emb_t = self.process_sigma_embeddings(sigma) if not self.use_ada_conditioning else None
        goals = self.preprocess_goals(goals, t, uncond)
        state_embed, proprio_states = self.process_state_embeddings(states)
        goal_embed = self.process_goal_embeddings(goals, states)
        action_embed = self.action_emb(actions)

        if self.use_abs_pos_emb:
            goal_x, state_x, action_x, proprio_x = self.apply_position_embeddings(goal_embed, state_embed, action_embed, proprio_states, t)
        else:
            goal_x = self.drop(goal_embed)
            state_x = self.drop(state_embed)
            action_x = self.drop(action_embed)
            proprio_x = self.drop(proprio_states) if proprio_states is not None else None

        input_seq = self.concatenate_inputs(emb_t, goal_x, state_x, action_x, proprio_x, uncond)
        context = self.encoder(input_seq)

        return context

    def process_goal_embeddings(self, goals, states):
        if self.use_modality_encoder and 'modality' in states and states['modality'] == 'lang':
            goal_embed = self.lang_emb(goals)
        else:
            goal_embed = self.goal_emb(goals)
        return goal_embed

    def process_sigma_embeddings(self, sigma):
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        return emb_t

    def preprocess_goals(self, goals, states_length,uncond=False):
        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[1] == states_length and self.goal_seq_len == 1:
            goals = goals[:, 0, :]
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[-1] == 2 * self.obs_dim:
            goals = goals[:, :, :self.obs_dim]
        if self.training:
            goals = self.mask_cond(goals)
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)
        return goals

    def process_state_embeddings(self, states):
        states_global = self.tok_emb(states['static'].to(torch.float32))
        incam_states = self.incam_embed(states['gripper'].to(torch.float32))
        proprio_states = None
        state_embed = torch.stack((states_global, incam_states), dim=2).reshape(states['gripper'].to(torch.float32).size(0), 2, self.embed_dim)
        # print(state_embed.shape)
        proprio_states = None
        return state_embed, proprio_states

    def apply_position_embeddings(self, goal_embed, state_embed, action_embed, proprio_states, t):
        position_embeddings = self.pos_emb
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, 1:, :])
        proprio_x = self.drop(proprio_states + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :]) if proprio_states is not None else None
        return goal_x, state_x, action_x, proprio_x

    def concatenate_inputs(self, emb_t, goal_x, state_x, action_x, proprio_x, uncond=False):
        if self.goal_conditioned:
            if self.use_ada_conditioning:
                input_seq = torch.cat([goal_x, state_x, proprio_x], dim=1) if proprio_x is not None else torch.cat([goal_x, state_x], dim=1)
            else:
                input_seq = torch.cat([emb_t, goal_x, state_x, proprio_x], dim=1) if proprio_x is not None else torch.cat([emb_t, goal_x, state_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x, proprio_x], dim=1) if proprio_x is not None else torch.cat([emb_t, state_x], dim=1)

        return input_seq
