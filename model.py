import torch
import torch.nn as nn
from typing import Optional

from layers import  CrossAttentionFusion
from streams import iTransformerStream, PowerformerStream



class Model(nn.Module):
    """
    Dual-stream time series forecasting architecture
    
    Input:  [batch, lookback_steps, num_variates]
    Output: [batch, forecast_steps, num_variates]
    
    Stream 1 (iTransformer): Treats each variate as a token, captures multivariate correlations
    Stream 2 (Powerformer): Patches the temporal dimension, captures temporal patterns
    Fusion: Cross-attention between the two streams
    """
    def __init__(
        self,
        # === Data Dimensions ===
        num_variates: int,          # Number of time series 
        lookback_steps: int,        # How many timesteps to look back 
        forecast_steps: int,        # How many timesteps to predict 
        
        # === Model Configurations ===
        skip_connections: bool = True, # Whether to use skip connections in the model
        d_model: int = 128, # NEED TO BE THE SAME FOR BOTH STREAMS FOR CROSS-ATTENTION
        # === Stream 1: iTransformer ===
        # Processes variates as tokens to capture multivariate relationships
        n_heads_s1: int = 8,        # Number of attention heads
        n_layers_s1: int = 3,       # Number of transformer layers
        d_ff_s1: int = 512,         # Feed-forward hidden dimension
        dropout_s1: float = 0.1,
        
        # === Stream 2: Powerformer ===
        # Patches time series to capture local temporal patterns
        patch_len: int = 16,        # Length of each patch 
        stride: int = 8,            # Stride between patches
        n_heads_s2: int = 8,        # Number of attention heads
        n_layers_s2: int = 3,       # Number of transformer layers
        d_ff_s2: int = 512,         # Feed-forward hidden dimension
        dropout_s2: float = 0.1,
        attn_decay_scale: float = 0.25,    # Decay rate (alpha)

        # === Fusion & Prediction ===
        n_heads_fusion: int = 8,
        use_fusion: bool = True,  # If False, concat streams directly instead of cross-attention
    ):
        super().__init__()
        
        # Store dimensions
        self.num_variates = num_variates
        self.lookback_steps = lookback_steps
        self.forecast_steps = forecast_steps
        self.skip_connections = skip_connections
        self.use_fusion = use_fusion

        # Skip connection: simple linear baseline
        if skip_connections:
            # Project from lookback to forecast dimension
            # [B, T, C] -> [B, F, C] via learned linear transform
            self.input_proj = nn.Linear(lookback_steps, forecast_steps)
            
            # Learnable scaling factor for residual
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        # ===== STREAM 1: iTransformer =====
        # Input:  [B, lookback_steps, num_variates]
        # Output: [B, num_variates, d_model]
        self.stream1 = iTransformerStream(
            num_variates=num_variates,
            lookback_steps=lookback_steps,
            d_model=d_model,
            n_heads=n_heads_s1,
            n_layers=n_layers_s1,
            d_ff=d_ff_s1,
            dropout=dropout_s1,
        )
        
        # ===== STREAM 2: Powerformer =====
        # Input:  [B, lookback_steps, num_variates]
        # Output: [B, num_variates, d_model]
        self.stream2 = PowerformerStream(
            num_variates=num_variates,
            lookback_steps=lookback_steps,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads_s2,
            n_layers=n_layers_s2,
            d_ff=d_ff_s2,
            dropout=dropout_s2,
            attn_decay_scale=attn_decay_scale,
        )
        
        # ===== FUSION & PREDICTION =====
        # Input:  [B, lookback_steps, num_variates]
        # Output: [B, num_variates, d_model]
        if use_fusion:
            self.fusion = CrossAttentionFusion(
                d_model=d_model,
                n_heads=n_heads_fusion,
                dropout=(dropout_s1 + dropout_s2) / 2,
            )
        
        # [B, C, 2*D] -> [B, C, forecast_steps]
        self.projection = nn.Linear(d_model * 2, forecast_steps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual streams
        
        Args:
            x: [batch, lookback_steps, num_variates]
            
        Returns:
            predictions: [batch, forecast_steps, num_variates]
        """
        B, T, C = x.shape  # batch, time, channels(variates)
        assert T == self.lookback_steps, f"Expected {self.lookback_steps} lookback steps, got {T}"
        assert C == self.num_variates, f"Expected {self.num_variates} variates, got {C}"

        # ===== SKIP CONNECTION (if enabled) =====
        if self.skip_connections:
            # Project input: [B, T, C] -> [B, C, T] -> [B, C, F]
            x_transposed = x.transpose(1, 2)  # [B, C, T]
            skip = self.input_proj(x_transposed)  # [B, C, F]
            skip = skip.transpose(1, 2)  # [B, F, C]
        
        # ===== STREAM 1: iTransformer =====
        # [B, T, C] -> [B, C, D]
        stream1_out = self.stream1(x)
        
        # ===== STREAM 2: Powerformer =====
        # [B, T, C] -> [B, C, D]
        stream2_out = self.stream2(x)
        
        # ===== FUSION =====
        # [B, C, D], [B, C, D] -> [B, C, 2*D]
        if self.use_fusion:
            fused = self.fusion(stream1_out, stream2_out)
        else:
            # Simple concatenation without cross-attention
            fused = torch.cat([stream1_out, stream2_out], dim=-1)
        
        # ===== PREDICT =====
        # [B, C, 2*D] -> [B, C, forecast_steps] -> [B, forecast_steps, C]
        predictions = self.projection(fused) 
        predictions = predictions.transpose(1, 2) 

        # ===== ADD SKIP CONNECTION =====
        if self.skip_connections:
            # Residual: predictions = model_output + scale * baseline
            predictions = predictions + self.residual_scale * skip

        return predictions 
    
    def forecast(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Forecast future timesteps
        
        Args:
            x: [batch, lookback_steps, num_variates]
            steps: Number of steps to forecast (default: forecast_steps)
            
        Returns:
            predictions: [batch, steps, num_variates]
        """
        if steps is None:
            steps = self.forecast_steps
        
        if steps <= self.forecast_steps:
            # Direct prediction
            return self.forward(x)[:, :steps, :]
        else:
            # TODO: Autoregressive forecasting for longer horizons
            return self.forward(x)


