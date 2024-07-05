from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor, to_pil_image


from voltron.models.util.transformer import Block, PatchEmbed, RMSNorm, get_2D_position_embeddings

# Helper/Utility Function -- computes simple 1D sinusoidal position embeddings for both 1D/2D use cases.
#   > We'll be combining two 1D sin-cos (traditional) position encodings for height/width of an image (grid features).
def get_1D_sine_cosine(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)  # [flatten(pos) x omega] -- outer product!
    emb_sin, emb_cos = np.sin(out), np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # [flatten(pos) x D]

# 1D Sine-Cosine Position Embedding -- standard from "Attention is all you need!"
def get_1D_position_embeddings(embed_dim: int, length: int) -> np.ndarray:
    return get_1D_sine_cosine(embed_dim, np.arange(length))

# 2D Sine-Cosine Position Embedding (from MAE repository)
#   > https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2D_position_embeddings(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    # Create 2D Position embeddings by taking cross product of height and width and splicing 1D embeddings...
    grid_h, grid_w = np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)  # w goes first?

    # Use half of dimensions to encode grid_h, other half to encode grid_w
    emb_h, emb_w = get_1D_sine_cosine(embed_dim // 2, grid[0]), get_1D_sine_cosine(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    # CLS token handling (only for R-MVP)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed
    
    
# Patch Embedding Module
class PatchEmbed(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        embed_dim: int,
        in_channels: int = 3,
        flatten: bool = True,
    ):
        super().__init__()
        self.resolution, self.patch_size = (resolution, resolution), (patch_size, patch_size)
        self.grid_size = (self.resolution[0] // self.patch_size[0], self.resolution[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.proj(patches)
        if self.flatten:
            return rearrange(patch_embeddings, "bsz embed patch_h patch_w -> bsz (patch_h patch_w) embed")
        return patch_embeddings
    
    
class MaskedTransformerImgDecoder(nn.Module):
    
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        decoder_depth: int,
        decoder_embed_dim: int,
        decoder_n_heads: int,
        context_dim: int,
        symmetric_mask: bool = True,
        num_images: int = 2,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        mask_ratio: float = 0.9,
        img_gen_frame_diff: int =3,
        video_gen: bool = False,
        norm_pixel_loss: bool = True
    ):
        super().__init__()
        self.img_gen_frame_diff = img_gen_frame_diff
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.resolution = resolution
        self.symmetric_mask = symmetric_mask
        self.num_patches = (resolution // patch_size) ** 2
        self.patch2embed = PatchEmbed(resolution, patch_size, decoder_embed_dim, in_channels=in_channels)
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.decoder_embed_dim, self.decoder_n_heads, self.decoder_depth = decoder_embed_dim, decoder_n_heads, decoder_depth
        # Projection from Encoder to Decoder
        self.encoder2decoder = nn.Linear(context_dim, self.decoder_embed_dim)
        self.num_images = num_images
        # MAE Decoder Parameters -- Remember the CLS Token (if specified)!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.ctx_dec_pe = nn.Parameter(torch.randn(1, 2, 1, self.decoder_embed_dim))
        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.num_patches, self.decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.decoder_embed_dim,
                    self.decoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = RMSNorm(self.decoder_embed_dim)
        self.decoder_patch_prediction = nn.Linear(self.decoder_embed_dim, (patch_size**2) * in_channels, bias=True)
        self.video_gen = video_gen
        self.initialize_weights()
    
    def mask(
        self, ctx_patches: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-context random masking by shuffling :: uses argsort random noise to identify masked patches."""
        bsz, ctx_len, n_patches, embed_dim = ctx_patches.shape
        if mask_ratio is not None:
            n_keep = int(n_patches * (1 - mask_ratio))
        else:
            n_keep = int(n_patches * (1 - self.mask_ratio))

        # Sample noise of n_patches size, argsort to get shuffled IDs, argsort again to get "unshuffle"
        #   > For clarity -- argsort is an invertible transformation (if argsort `restore`, recovers `shuffle`)
        #   > Note that shuffle_idxs is defined solely as a function of *n_patches* and **not** context! Same mask!
        # Get "keep" (visible) patches --> make sure to get _same_ patches *across* context length!
        if self.symmetric_mask:
            shuffle_idxs = torch.argsort(torch.rand(bsz, n_patches, device=ctx_patches.device), dim=1)
            restore_idxs = torch.argsort(shuffle_idxs, dim=1)
            visible_patches = torch.gather(
                ctx_patches, dim=2, index=shuffle_idxs[:, None, :n_keep, None].repeat(1, ctx_len, 1, embed_dim)
            )
            # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following FAIR MAE convention)
            mask = torch.ones(bsz, n_patches, device=ctx_patches.device)
            mask[:, :n_keep] = 0
            mask = torch.gather(mask, dim=1, index=restore_idxs)
        else:
            shuffle_idxs = torch.argsort(torch.rand(bsz,ctx_len, n_patches, device=ctx_patches.device), dim=1)
            restore_idxs = torch.argsort(shuffle_idxs, dim=1)
            visible_patches = torch.gather(
                ctx_patches, dim=2, index=shuffle_idxs[:, :, :n_keep, None].repeat(1, 1, 1, embed_dim)
            )
            # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following FAIR MAE convention)
            mask = torch.ones(bsz, ctx_len, n_patches, device=ctx_patches.device)
            mask[:, :, :n_keep] = 0
            mask = torch.gather(mask, dim=1, index=restore_idxs)
             # Change the shape of restore_idxs to match the requirements of the forward method
            # Change the shape of restore_idxs to match the requirements of the forward method
            # restore_idxs = restore_idxs.unsqueeze(1).expand(1, ctx_len, -1)


        return visible_patches, mask, restore_idxs



    def initialize_weights(self) -> None:

        dec_pe = get_2D_position_embeddings(
            self.decoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=False
        )
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize Mask Token, Img Token, Lang Token w/ Truncated Normal
        nn.init.normal_(self.mask_token, std=0.02)
        # Everything else...
        self.apply(self.transformer_initializer)
    
    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # Use xavier_uniform following Jax ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of (0th + Kth frame) images to their patched equivalents by naive reshaping."""
        return rearrange(
            imgs,
            "bsz ctx c (height patch_h) (width patch_w) -> bsz ctx (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )
    
    def forward(self, context, target_images, img_gen_frame_diff: int =3):
        """
        Forward method for the Transformer Image Decoder.
        
        Args:
            context (torch.Tensor): Context embeddings from the encoder.
            masked_patches (torch.Tensor): Masked patches to reconsruct with the

        Returns:
            torch.Tensor: Reconstructed image patches.
        """
        # Step 1: Project the context from the encoder to the decoder's dimension
        emb_context = self.encoder2decoder(context)
        
        # Step 2: Generate tokens for all patches of the image and mask them
        patches = self.patch2embed(rearrange(target_images, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        # add position embeddings to all tokens
        ctx_patches_pe = patches + self.decoder_pe.to(context.device)
        ctx_patches = rearrange(ctx_patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=self.num_images)
        # Create mask (and go ahead and mask out patches at the same time)
        visible_ctx_patches, mask, restore_idxs = self.mask(ctx_patches, self.mask_ratio)
        
        visible_patches = rearrange(visible_ctx_patches, "bsz ctx seq embed -> bsz (ctx seq) embed")
        
        visible_per_frame = (visible_patches.shape[1]) // 2
        
        # Add Mask Tokens to Sequence and Unshuffle
        if self.symmetric_mask:
            mask_tokens = self.mask_token.repeat(visible_patches.shape[0], 2, restore_idxs.shape[1] - visible_per_frame, 1)
        else:
            mask_tokens = self.mask_token.repeat(visible_patches.shape[0], 2, restore_idxs.shape[2] - visible_per_frame, 1)
        
        projected_ctx_patches = rearrange(visible_patches, "bsz (ctx seq) embed -> bsz ctx seq embed", ctx=self.num_images)
        concatenated_ctx_patches = torch.cat([projected_ctx_patches, mask_tokens], dim=2)
        if self.symmetric_mask:
            unshuffled_ctx_patches = torch.gather(
                concatenated_ctx_patches,
                dim=2,
                index=restore_idxs[:, None, ..., None].repeat(1, 2, 1, self.decoder_embed_dim),
            )
        else:
            unshuffled_ctx_patches = torch.gather(
                concatenated_ctx_patches,
                dim=2,
                index=restore_idxs.unsqueeze(-1).repeat(1, 1, 1, self.decoder_embed_dim)
            )

        
        # Add position embeddings, `ctx_dec_pe` embeddings, and flatten patches for Transformer...
        # tells the model which path belongs to which image
        decoder_ctx_patches_pe = unshuffled_ctx_patches + (
            self.decoder_pe[None, ...]
        )
        decoder_ctx_patches = decoder_ctx_patches_pe + self.ctx_dec_pe[:, :2, ...]
        decoder_patches = rearrange(decoder_ctx_patches, "bsz ctx seq embed -> bsz (ctx seq) embed")
        
        # add context embeddings
        decoder_patches = torch.cat([emb_context, decoder_patches], dim=1)
        # Step 3: Pass the combined input through transformer decoder blocks
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        tokens = self.decoder_norm(decoder_patches)

        # Step 4: Predict the reconstructed patches
        # (assuming the first `N` tokens correspond to the original context)
        reconstructions = self.decoder_patch_prediction(tokens[:, context.size(1):])
        # reshape
        reconstructions = rearrange(reconstructions, 'b (t n) d -> b t n d', t=self.num_images)
        return reconstructions, mask, restore_idxs, visible_patches

    def compute_loss(
        self,
        imgs: torch.Tensor,
        ctx_reconstructions: torch.Tensor,
        mask: torch.Tensor,
        restore_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction Loss...
        assert self.norm_pixel_loss, "`norm_pixel_loss` should always be true... false only for visualizations!"
        targets = self.patchify(imgs)

        # Split targets into 0 and K --> do the same for ctx_reconstructions
        zero_target, k_target = targets[:, 0, ...], targets[:, 1, ...]
        zero_reconstruction, k_reconstruction = ctx_reconstructions[:, 0, ...], ctx_reconstructions[:, 1, ...]

        # Compute mean losses per patch first...
        zero_mse, k_mse = (zero_reconstruction - zero_target) ** 2, (k_reconstruction - k_target) ** 2
        zero_avg_loss_per_patch, k_avg_loss_per_patch = zero_mse.mean(dim=-1), k_mse.mean(dim=-1)

        # Compute reconstruction losses...
        if self.symmetric_mask:
            zero_loss = (zero_avg_loss_per_patch * mask).sum() / mask.sum() 
            k_loss = (k_avg_loss_per_patch * mask).sum() / mask.sum()
        else:
            zero_loss = (zero_avg_loss_per_patch * mask[:, 0]).sum() / mask[:, 0].sum() 
            k_loss = (k_avg_loss_per_patch * mask[:, 1]).sum() / mask[:, 1].sum()
        reconstruction_loss = (zero_loss + k_loss) / 2

        return reconstruction_loss
    
    def reconstruct_image(
        self, 
        predictions: torch.Tensor, 
        goal_images: torch.Tensor, 
        mask: torch.Tensor,
        restore_idxs: torch.Tensor, 
        file_path: str
    ) -> None:
        """
        Visualize the reconstructed images and save them side by side as a PNG file.

        Args:
            predictions (torch.Tensor): The reconstructed image patches tensor output by the decoder.
                                       Shape: [batch_size, num_images, num_pred_patches, d]
            file_path (str): The path where the PNG file will be saved.
        """
        # Normalization values
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

        # Reshape goal_images to match the structure of predictions
        goal_img_patches = self.patchify(goal_images)


        # Extract the two images from the predictions
        bsz, num_images, num_patches, d = predictions.size()
        assert num_images == self.num_images, "Number of images should be 2"

        pil_images = []
        for i in range(num_images):
            # Reshape the reconstructed patches to the original image layout
            patch_h, patch_w = self.patch_size, self.patch_size
            grid_h, grid_w = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
            
            # Combine visible and reconstructed patches
            recon_patches = predictions[:, i].to(torch.float32)
            visible_patches = goal_img_patches[:, i]
            combined_patches = visible_patches.clone()

            # Expand restore_idxs to match the dimensions of combined_patches
            expanded_restore_idxs = restore_idxs.unsqueeze(-1).expand(-1, -1, recon_patches.shape[-1])

            # Reorder patches to original positions using restore_idxs
            reordered_patches = torch.gather(recon_patches, 1, expanded_restore_idxs)
            
            # Use mask to select patches
            visible_patches[mask == 1] = reordered_patches[mask == 1]
            # Reshape patches back to image
            recon_img = visible_patches.view(bsz, grid_h, grid_w, patch_h, patch_w, self.in_channels)
            recon_img = recon_img.permute(0, 5, 1, 3, 2, 4)
            recon_img = recon_img.reshape(bsz, self.in_channels, grid_h * patch_h, grid_w * patch_w)

            # Denormalize and convert to PIL image
            single_img = denormalize(recon_img[0], mean, std)
            single_img = single_img.to(torch.float32).detach().cpu().clamp(0, 1)
            single_img = (single_img * 255).to(torch.uint8)
            pil_images.append(Image.fromarray(single_img.permute(1, 2, 0).numpy(), 'RGB'))

        # Concatenate images side by side
        total_width = pil_images[0].width * num_images
        total_height = pil_images[0].height
        combined_img = Image.new('RGB', (total_width, total_height))

        x_offset = 0
        for im in pil_images:
            combined_img.paste(im, (x_offset, 0))
            x_offset += im.width

        combined_img.save(file_path, 'PNG')
        print(f"Image saved to {file_path}")
        
        

def denormalize(tensor, means, stds):
    """
    Denormalize a tensor image with mean and standard deviation.
    Args:
        tensor (Tensor): Normalized image tensor.
        means (list): Means for each channel.
        stds (list): Standard deviations for each channel.
    Returns:
        Tensor: Denormalized image tensor.
    """
    denormalized = tensor.clone()
    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized