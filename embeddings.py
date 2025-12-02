import torch
import torch.nn as nn
import math

class VariateEmbedding(nn.Module):
    """
    embeds entire time series as a variate token
    
    Input:  [B, T, C] where T=seq_len, C=num_variates
    Output: [B, C, D] where D=d_model
    """
    
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # MLP: R^T -> R^D for each variate
        self.embedding = nn.Sequential(
            nn.Linear(seq_len, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            [B, C, D] - variate tokens
        """
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.embedding(x)  # [B, C, D]
        return x


class PatchEmbedding(nn.Module):
    """
    1D convolutional patch embedding
    
    Input:  [B, T, C] where T=seq_len, C=num_variates
    Output: [B, C, N, D] where N=num_patches, D=d_model
    """
    
    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Conv1d for patching
        self.patch_proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
            bias=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            [B, C, N, D] - patch embeddings per variate
        """
        B, T, C = x.shape
        
        # Process each variate independently
        x = x.transpose(1, 2)  # [B, C, T]
        x = x.reshape(B * C, 1, T)  # [B*C, 1, T]
        
        # Apply patching
        x = self.patch_proj(x)  # [B*C, D, N]
        
        # Reshape back
        N = x.shape[-1]
        x = x.transpose(1, 2)  # [B*C, N, D]
        x = x.reshape(B, C, N, self.d_model)  # [B, C, N, D]
        
        x = self.dropout(x)
        
        return x
    
    def get_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches"""
        return (seq_len - self.patch_len) // self.stride + 1


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for patch sequences
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N, D] or [B, N, D]
        Returns:
            Same shape with positional encoding added
        """
        if x.dim() == 4:  # [B, C, N, D]
            x = x + self.pe[:x.size(2), :].unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:  # [B, N, D]
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        
        return self.dropout(x)
    