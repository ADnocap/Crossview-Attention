import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N=num_tokens, D=d_model
            mask: Optional [B, N, N] or [N, N] attention mask
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        
        # Linear projections and split into heads
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, N, N]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ V  # [B, H, N, d_k]
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        out = self.W_o(out)
        
        return out


class WeightedCausalAttention(nn.Module):
    """
    Weighted Causal Multi-head Attention (WCMHA) with power-law decay
    Applies causal mask + power-law decay to attention scores
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        decay_scale: float = 0.25  # alpha parameter
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.decay_scale = decay_scale
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # No dropout on attention weights (per Powerformer paper)
        # Dropout conflicts with power-law bias
        
    def _create_power_law_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create power-law decay mask: f(Δt) = -α * log(Δt)
        
        Returns:
            [seq_len, seq_len] mask where M[i,j] = -α*log(i-j) for j<=i, -inf for j>i
        """
        i = torch.arange(seq_len, device=device).view(-1, 1)
        j = torch.arange(seq_len, device=device).view(1, -1)
        
        delta_t = (i - j).float()
        mask = torch.zeros_like(delta_t)
        
        # Apply power-law decay only to valid positions
        valid = delta_t > 0
        mask[valid] = -self.decay_scale * torch.log(delta_t[valid])
        
        # Causal mask
        mask = mask.masked_fill(j > i, float('-inf'))
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N=num_patches, D=d_model
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        
        # Linear projections and split into heads
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        # Scaled dot-product attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, N, N]
        
        # Create and apply power-law causal mask
        mask = self._create_power_law_mask(N, x.device)  # [N, N]
        scores = scores + mask.unsqueeze(0).unsqueeze(0)  # Broadcast to [B, H, N, N]
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, N]
        
        # Apply attention to values
        out = attn_weights @ V  # [B, H, N, d_k]
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        out = self.W_o(out)
        
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between two streams
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Cross-attention: stream1 -> stream2
        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_k2 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)
        
        # Cross-attention: stream2 -> stream1
        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_k1 = nn.Linear(d_model, d_model)
        self.W_v1 = nn.Linear(d_model, d_model)
        
        # Output projections
        self.W_o1 = nn.Linear(d_model, d_model)
        self.W_o2 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, stream1: torch.Tensor, stream2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stream1: [B, C, D] - iTransformer features
            stream2: [B, C, D] - Powerformer features
            
        Returns:
            fused: [B, C, D*2] - combined features
        """
        B, C, D = stream1.shape
        
        # === Cross-attention: stream1 attends to stream2 ===
        Q1 = self.W_q1(stream1).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        K2 = self.W_k2(stream2).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        V2 = self.W_v2(stream2).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        
        scores1 = torch.matmul(Q1, K2.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn1 = F.softmax(scores1, dim=-1)
        attn1 = self.dropout(attn1)
        
        out1 = torch.matmul(attn1, V2)  # [B, H, C, d_k]
        out1 = out1.transpose(1, 2).contiguous().view(B, C, D)
        out1 = self.W_o1(out1)  # [B, C, D]
        
        # === Cross-attention: stream2 attends to stream1 ===
        Q2 = self.W_q2(stream2).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        K1 = self.W_k1(stream1).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        V1 = self.W_v1(stream1).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        
        scores2 = torch.matmul(Q2, K1.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn2 = F.softmax(scores2, dim=-1)
        attn2 = self.dropout(attn2)
        
        out2 = torch.matmul(attn2, V1)  # [B, H, C, d_k]
        out2 = out2.transpose(1, 2).contiguous().view(B, C, D)
        out2 = self.W_o2(out2)  # [B, C, D]
        
        # Concatenate both cross-attention outputs
        combined = torch.cat([out1, out2], dim=-1)  # [B, C, 2*D]
        
        return combined  # [B, C, 2*D]