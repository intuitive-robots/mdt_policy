# This code is referenced from https://github.com/dhansmair/flamingo-mini

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

from .utils import feed_forward_layer


class PerceiverAttentionLayer(nn.Module):
    """Perceiver Attention Layer"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents):
        """Latent vectors are cross-attending to the visual features x

        Args:
            features: Batch of visual features with shape (batch_size, n_features, dim)
            latents: Latent learnt vectors which are used to compute queries with shape (batch_size, n_latents, dim)

        Returns:
            Attention score with shape (batch_size, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # Layer normalization
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # Compute the queries from the latents, for all attention heads simultaneously
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # Keys and values for all attention heads
        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=n_heads)
        assert v.shape == torch.Size([n_batch, n_heads, n_features_latents, self.dim_head])

        q = q * self.scale

        # Attention scores
        sim = einsum('b h q d, b h f d -> b h q f', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum('b h q f, b h f v -> b h q v', alphas, v)
        out = rearrange(out, 'b h q v -> b q (h v)')

        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler with multi-head attention layer"""

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_time_embeds: int = 4,
        ff_mult: int = 4,
        activation: str = 'gelu',
        trainable: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, dim))  # type: ignore[reportPrivateUsage]
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))  # type: ignore[reportPrivateUsage]

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                        feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                    ]
                )
            )

        # Layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(dim)

        self._update_trainable_state(trainable)
        
    def _update_trainable_state(self, trainable: bool = True):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x_f: torch.Tensor, mask: torch.BoolTensor = None):
        """Run perceiver resampler on the input visual embeddings

        Args:
            x_f: Input visual embeddings of shape (batch_size, n_frames, n_features, d_visual)
            mask: Mask for the input visual embeddings of shape (batch_size, n_frames)

        Returns:
            Resampler features of shape (batch_size, num_queries, d_visual)
        """
        assert x_f.ndim == 4

        batch_size, max_length, _, dim = x_f.shape

        assert dim == self.dim

        # Mask the position embeddings for the padded frames
        time_pos_emb = (
            self.time_pos_emb[:max_length].unsqueeze(0).expand(batch_size, -1, -1, -1)
        )  # [batch_size, max_length, 1, dim]
        if mask is not None:
            time_pos_emb = time_pos_emb * mask.unsqueeze(-1).unsqueeze(-1)

        # Apply the position embeddings
        x_f = x_f + time_pos_emb

        # Flatten the frames
        x_f = rearrange(x_f, 'b T n d -> b (T n) d')

        # Copy the latents for every element in the batch
        x = repeat(self.latents, 'q d -> b q d', b=batch_size)

        # Apply attention and feed forward layer
        for attn, ffw in self.layers:
            x = x + attn(x_f, x)
            x = x + ffw(x)

        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])

        norm = self.norm(x)
        return norm