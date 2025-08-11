"""
GQA Transformer model for stock price prediction
Implements Grouped-Query Attention for efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import config


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention implementation
    Reduces KV heads while maintaining query heads for efficiency
    """
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.group_size = num_heads // num_kv_heads
        
        # Linear projections - all use same head_dim for consistency
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(model_dim, self.head_dim * num_kv_heads, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, model_dim]
        k = self.k_proj(x)  # [batch, seq_len, head_dim * num_kv_heads]
        v = self.v_proj(x)  # [batch, seq_len, head_dim * num_kv_heads]
        
        # Reshape Q
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape K, V
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Repeat K, V for each group
        k = k.repeat_interleave(self.group_size, dim=1)  # [batch, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(self.group_size, dim=1)  # [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        out = self.out_proj(out)
        
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series data
    """
    def __init__(self, model_dim: int, max_len: int = config.MAX_SEQUENCE_LENGTH):
        super().__init__()
        
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * 
                           (-math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with GQA
    """
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int, 
                 ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = GroupedQueryAttention(model_dim, num_heads, num_kv_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class StockPredictionTransformer(nn.Module):
    """
    Main transformer model for stock price prediction with uncertainty quantification
    """
    def __init__(self, input_dim: int, model_dim: int = config.MODEL_DIM, 
                 num_heads: int = config.NUM_HEADS, num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT_RATE):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_kv_heads = config.GQA_NUM_KEY_VALUE_HEADS
        self.dropout_rate = dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_dim)
        
        # Transformer encoder layers
        ff_dim = model_dim * 4  # Standard transformer feed-forward dimension
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, self.num_kv_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers for mean prediction
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)  # Single value prediction
        )
        
        # Additional output layer for uncertainty estimation (log variance)
        self.uncertainty_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)  # Log variance prediction
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize model weights
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                return_uncertainty: bool = False) -> torch.Tensor:
        """
        Forward pass with optional uncertainty estimation
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            return_uncertainty: If True, returns (prediction, log_variance)
        
        Returns:
            If return_uncertainty=False: Predicted close price [batch_size, 1]
            If return_uncertainty=True: Tuple of (prediction, log_variance)
        """
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, model_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch, model_dim]
        
        # Output projection for mean prediction
        prediction = self.output_projection(x)  # [batch, 1]
        
        if return_uncertainty:
            # Output projection for uncertainty (log variance)
            log_variance = self.uncertainty_projection(x)  # [batch, 1]
            return prediction, log_variance
        
        return prediction
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100, 
                                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty estimation
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            n_samples: Number of forward passes for MC Dropout
            mask: Optional attention mask
            
        Returns:
            Tuple of (mean_prediction, prediction_variance)
        """
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x, mask, return_uncertainty=False)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [n_samples, batch_size, 1]
        
        mean_prediction = predictions.mean(dim=0)
        prediction_variance = predictions.var(dim=0)
        
        self.eval()  # Switch back to eval mode
        
        return mean_prediction, prediction_variance
    
    def get_prediction_interval(self, x: torch.Tensor, confidence: float = 0.95,
                               n_samples: int = 100, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prediction confidence intervals
        
        Args:
            x: Input tensor
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            n_samples: Number of MC dropout samples
            mask: Optional attention mask
            
        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound)
        """
        mean_pred, variance = self.predict_with_uncertainty(x, n_samples, mask)
        std_dev = torch.sqrt(variance)
        
        # Calculate confidence interval using normal approximation
        from scipy.stats import norm
        alpha = 1 - confidence
        z_score = norm.ppf(1 - alpha/2)
        
        lower_bound = mean_pred - z_score * std_dev
        upper_bound = mean_pred + z_score * std_dev
        
        return mean_pred, lower_bound, upper_bound
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_dim: int) -> StockPredictionTransformer:
    """
    Create and initialize the model
    """
    model = StockPredictionTransformer(input_dim)
    
    # Move to device if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Created model with {model.count_parameters():,} parameters")
    print(f"Model device: {device}")
    
    return model


def main():
    """
    Test model creation and forward pass
    """
    # Assume 10 input features (5 target + 5 market features)
    input_dim = 10
    model = create_model(input_dim)
    
    # Test forward pass
    batch_size = 4
    seq_len = config.SEQUENCE_LENGTH
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample predictions: {output.squeeze().tolist()}")


if __name__ == "__main__":
    main()