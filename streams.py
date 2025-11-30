import torch
import torch.nn as nn

from layers import FeedForward, MultiHeadAttention, WeightedCausalAttention
from embeddings import VariateEmbedding, PatchEmbedding, PositionalEncoding

class iTransformerStream(nn.Module):
    def __init__(
        self,
        num_variates: int,
        lookback_steps: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Embedding
        self.embedding = VariateEmbedding(lookback_steps, d_model, dropout)
        
        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            # Attention + residual + norm
            attn = MultiHeadAttention(d_model, n_heads, dropout)
            ffn = FeedForward(d_model, d_ff, dropout)
            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            dropout1 = nn.Dropout(dropout)
            dropout2 = nn.Dropout(dropout)
            
            self.encoder_layers.append(nn.ModuleDict({
                'attn': attn,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
                'dropout1': dropout1,
                'dropout2': dropout2,
            }))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed: [B, T, C] -> [B, C, D]
        x = self.embedding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            # Attention with residual
            attn_out = layer['attn'](x)
            x = x + layer['dropout1'](attn_out)
            x = layer['norm1'](x)
            
            # FFN with residual
            ffn_out = layer['ffn'](x)
            x = x + layer['dropout2'](ffn_out)
            x = layer['norm2'](x)
        
        return x  # [B, C, D]


class PowerformerStream(nn.Module):
    def __init__(
        self,
        num_variates: int,
        lookback_steps: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn_decay_scale: float = 0.25,
    ):
        super().__init__()
        
        self.c_in = num_variates
        
        # Patch embedding + positional encoding
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, dropout)
        self.pos_encode = PositionalEncoding(d_model, max_len=1000, dropout=dropout)
        
        # Number of patches
        self.num_patches = self.patch_embed.get_num_patches(lookback_steps)
        
        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            attn = WeightedCausalAttention(d_model, n_heads, attn_decay_scale, dropout)
            ffn = FeedForward(d_model, d_ff, dropout)
            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            dropout1 = nn.Dropout(dropout)
            dropout2 = nn.Dropout(dropout)
            
            self.encoder_layers.append(nn.ModuleDict({
                'attn': attn,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
                'dropout1': dropout1,
                'dropout2': dropout2,
            }))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Patch embedding: [B, T, C] -> [B, C, N, D]
        x = self.patch_embed(x)
        x = self.pos_encode(x)
        
        # Process each variate independently: [B, C, N, D] -> [B*C, N, D]
        x = x.view(B * C, self.num_patches, -1)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            # WCMHA with residual
            attn_out = layer['attn'](x)
            x = x + layer['dropout1'](attn_out)
            x = layer['norm1'](x)
            
            # FFN with residual
            ffn_out = layer['ffn'](x)
            x = x + layer['dropout2'](ffn_out)
            x = layer['norm2'](x)
        
        # Pool over patches: [B*C, N, D] -> [B*C, D]
        x = x.mean(dim=1)
        
        # Reshape back: [B*C, D] -> [B, C, D]
        x = x.view(B, C, -1)
        
        return x  # [B, C, D]

