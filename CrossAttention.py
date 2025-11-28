"""
Dual-Stream Time Series Forecasting Model with Gated Cross-Attention

Stream 1: iTransformer-style (variate tokens, attention across variates)
Stream 2: Powerformer-style (patch-based temporal modeling with power-law decay)
Gated Cross-Attention: Filters noisy cross-modal interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Model(nn.Module):
    """
    Dual-stream architecture combining:
    - Stream 1: iTransformer (variate-centric modeling)
    - Stream 2: Powerformer (patch-based temporal modeling)
    - Gated Cross-Attention between streams
    """
    
    def __init__(
        self,
        # Data dimensions
        c_in: int,                      # Number of input variates
        context_window: int,            # Lookback window length
        target_window: int,             # Prediction window length
        
        # Stream 1 (iTransformer) params
        d_model_s1: int = 128, # Dimension of variate embeddings
        n_heads_s1: int = 8, # Number of attention heads
        n_layers_s1: int = 3, # Number of encoder layers
        d_ff_s1: int = 512, # Dimension of feed-forward network
        dropout_s1: float = 0.1, # Dropout rate
        
        # Stream 2 (Powerformer) params
        patch_len: int = 16, # Length of each patch
        stride: int = 8, # Stride between patches 
        d_model_s2: int = 128, # Dimension of patch embeddings
        n_heads_s2: int = 8, # Number of attention heads
        n_layers_s2: int = 3, # Number of encoder layers
        d_ff_s2: int = 512, # Dimension of feed-forward network
        dropout_s2: float = 0.1, # Dropout rate
        attn_decay_type: Optional[str] = "power",
        attn_decay_scale: float = 0.25,
        
        # Cross-attention params
        d_model_cross: int = 128, # Dimension of cross-attention output
        n_heads_cross: int = 8, # Number of attention heads
        gate_activation: str = "sigmoid",  # sigmoid or tanh
        
        # General params
        revin: bool = True, 
        affine: bool = True,
        subtract_last: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        
        # RevIN for both streams
        self.revin = revin
        if self.revin:
            # TODO: Implement or import RevIN layer
            # self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
            pass
        
        # ===== STREAM 1: iTransformer (Variate-centric) =====
        self.stream1 = iTransformerStream(
            c_in=c_in,
            seq_len=context_window,
            d_model=d_model_s1,
            n_heads=n_heads_s1,
            n_layers=n_layers_s1,
            d_ff=d_ff_s1,
            dropout=dropout_s1,
        )
        
        # ===== STREAM 2: Powerformer (Patch-based temporal) =====
        self.stream2 = PowerformerStream(
            c_in=c_in,
            context_window=context_window,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model_s2,
            n_heads=n_heads_s2,
            n_layers=n_layers_s2,
            d_ff=d_ff_s2,
            dropout=dropout_s2,
            attn_decay_type=attn_decay_type,
            attn_decay_scale=attn_decay_scale,
        )
        
        # ===== GATED CROSS-ATTENTION =====
        # S1 -> S2 (iTransformer features attending to Powerformer features)
        self.gated_cross_attn_s1_to_s2 = GatedCrossAttention(
            d_model_q=d_model_s1,
            d_model_kv=d_model_s2,
            d_model_out=d_model_cross,
            n_heads=n_heads_cross,
            gate_activation=gate_activation,
            dropout=dropout_s1,
        )
        
        # S2 -> S1 (Powerformer features attending to iTransformer features)
        self.gated_cross_attn_s2_to_s1 = GatedCrossAttention(
            d_model_q=d_model_s2,
            d_model_kv=d_model_s1,
            d_model_out=d_model_cross,
            n_heads=n_heads_cross,
            gate_activation=gate_activation,
            dropout=dropout_s2,
        )
        
        # ===== FUSION & PREDICTION HEAD =====
        # TODO: Implement fusion strategy
        self.fusion_layer = None  # TODO
        
        # Final combination head
        self.final_head = PredictionHead(
            d_model=d_model_cross * 2,
            c_out=c_in,
            pred_len=target_window,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual streams with gated cross-attention
        
        Args:
            x: Input tensor [batch_size, context_window, c_in]
            
        Returns:
            predictions: [batch_size, target_window, c_in]
        """
        # x shape: [B, T, C] where T=context_window, C=c_in
        
        # ===== NORMALIZATION =====
        if self.revin:
            # TODO: Apply RevIN normalization
            # x_norm = self.revin_layer(x, mode='norm')
            x_norm = x
        else:
            x_norm = x
        
        # ===== STREAM 1: iTransformer =====
        # Returns: [B, C, D] - variate tokens with embeddings
        stream1_features = self.stream1(x_norm)  # [B, C, d_model_s1]
        
        # ===== STREAM 2: Powerformer =====
        # Returns: [B, C, D] - patch-based features per variate
        stream2_features = self.stream2(x_norm)  # [B, C, d_model_s2]
        
        # ===== GATED CROSS-ATTENTION =====
        # Stream 1 attends to Stream 2 (with gating to filter noise)
        s1_cross_attended = self.gated_cross_attn_s1_to_s2(
            query=stream1_features,
            key=stream2_features,
            value=stream2_features,
        )  # [B, C, d_model_cross]
        
        # Stream 2 attends to Stream 1 (with gating to filter noise)
        s2_cross_attended = self.gated_cross_attn_s2_to_s1(
            query=stream2_features,
            key=stream1_features,
            value=stream1_features,
        )  # [B, C, d_model_cross]
        
        # ===== FUSION =====
        # TODO: Implement fusion
        fused_features = self.fusion_layer() # TODO  # [B, C, 2*d_model_cross]
        
        # ===== PREDICTION =====
        predictions = self.final_head(fused_features)  # [B, C, target_window]
        
        # Transpose to [B, target_window, C]
        predictions = predictions.transpose(1, 2)
        
        # ===== DENORMALIZATION =====
        if self.revin:
            # TODO: Apply RevIN denormalization
            # predictions = self.revin_layer(predictions, mode='denorm')
            pass
        
        return predictions
    
    def forecast(
        self, 
        x: torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Multi-step forecasting with optional autoregressive extension
        
        Args:
            x: Input tensor [batch_size, context_window, c_in]
            steps: Number of steps to forecast (default: target_window)
            
        Returns:
            predictions: [batch_size, steps, c_in]
        """
        if steps is None:
            steps = self.target_window
        
        if steps <= self.target_window:
            # Direct prediction
            return self.forward(x)[:, :steps, :]
        else:
            # TODO: Implement autoregressive forecasting for steps > target_window
            # For now, just return forward pass
            return self.forward(x)


class iTransformerStream(nn.Module):
    """
    Stream 1: iTransformer-style processing
    - Embeds each variate as a token (whole time series)
    - Attention captures multivariate correlations
    - FFN learns series representations
    """
    
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        # TODO
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            features: [B, C, D] - variate tokens with embeddings
        """
        # TODO
        pass


class iTransformerEncoderLayer(nn.Module):
    """Encoder layer for iTransformer"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # TODO
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D]
        Returns:
            x: [B, C, D]
        """
        # TODO
        pass


class PowerformerStream(nn.Module):
    """
    Stream 2: Powerformer-style processing
    - Patches time series
    - Attention with power-law decay masks
    - Channel-independent processing
    """
    
    def __init__(
        self,
        c_in: int,
        context_window: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn_decay_type: Optional[str] = "power",
        attn_decay_scale: float = 0.25,
    ):
        super().__init__()
        # TODO
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            features: [B, C, D]
        """
        # TODO
        pass


class PowerformerEncoderLayer(nn.Module):
    """Encoder layer for Powerformer with power-law attention decay"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        patch_num: int = 0,
        attn_decay_type: Optional[str] = "power",
        attn_decay_scale: float = 0.25,
    ):
        super().__init__()
        # TODO
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N = patch_num
        Returns:
            x: [B, N, D]
        """
        # TODO:
        pass
    
    # TODO: Implement power-law decay mask creation
    def _create_decay_mask(self):
        pass


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention Mechanism
    
    Filters noisy and non-meaningful cross-modal interactions using a learned gate.
    The gate determines how much information from the cross-stream should be incorporated.
    
    Key idea: In the absence of useful cross-modal interactions, the gate suppresses
    the cross-attention output, preventing noise from degrading the feature representation.
    """
    
    def __init__(
        self,
        d_model_q: int,        # Query dimension
        d_model_kv: int,       # Key/Value dimension
        d_model_out: int,      # Output dimension
        n_heads: int = 8,
        gate_activation: str = "sigmoid",  # 'sigmoid' or 'tanh'
        dropout: float = 0.1,
    ):
        super().__init__()
        #TODO
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D_q] - features from one stream
            key: [B, N_k, D_k] - features from other stream
            value: [B, N_v, D_v] - features from other stream
            attn_mask: Optional attention mask
            
        Returns:
            gated_output: [B, N_q, D_out] - gated cross-attended features
        """
        #TODO
        pass

class PredictionHead(nn.Module):
    """Prediction head to map features to forecasts"""
    
    def __init__(
        self,
        d_model: int,
        c_out: int,
        pred_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # TODO
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D]
        Returns:
            predictions: [B, C, pred_len]
        """
        # TODO
        pass
