"""
FastSpeech 2 model for voice cloning.
Implements non-autoregressive TTS with duration and pitch control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class FastSpeech2Config:
    """Configuration for FastSpeech 2."""
    # Model dimensions
    encoder_embed_dim: int = 256
    encoder_num_layers: int = 4
    encoder_num_heads: int = 4
    encoder_ff_dim: int = 1024
    
    decoder_embed_dim: int = 256
    decoder_num_layers: int = 4
    decoder_num_heads: int = 4
    decoder_ff_dim: int = 1024
    
    # Text processing
    vocab_size: int = 300
    max_text_length: int = 200
    
    # Audio processing
    num_mels: int = 80
    max_mel_length: int = 1024
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sample_rate: int = 22050
    
    # Speaker embedding
    num_speakers: int = 100
    speaker_embed_dim: int = 192
    
    # Training
    dropout: float = 0.1
    length_predictor_kernel: int = 3
    
    # Variance predictor
    variance_predictor_kernel: int = 3
    pitch_quantization: str = "log"
    energy_quantization: str = "linear"
    
    # Initialization
    initializer_range: float = 0.02


class VariancePredictor(nn.Module):
    """
    Predicts duration, pitch, and energy.
    Used for controlling prosody in speech synthesis.
    
    Args:
        embed_dim: Input embedding dimension
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.proj = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict variance.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Attention mask (batch, seq_len)
            
        Returns:
            Predictions (batch, seq_len, 1)
        """
        # Transpose for conv
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Project to single value
        x = self.proj(x)
        
        # Remove padding
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        
        return x.squeeze(-1)  # (batch, seq_len)


class LengthRegulator(nn.Module):
    """
    Length regulator for duration-controlled upsampling.
    
    Args:
        embed_dim: Embedding dimension
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        duration_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Regulate length based on predicted duration.
        
        Args:
            x: Input sequence (batch, seq_len, embed_dim)
            duration: Duration predictions (batch, seq_len)
            duration_mask: Mask for valid positions
            
        Returns:
            Tuple of (expanded sequence, output mask)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Expand x according to duration
        output = []
        output_lengths = []
        
        for b in range(batch_size):
            if duration_mask is not None and not duration_mask[b].any():
                continue
                
            x_b = x[b]
            dur_b = duration[b]
            
            # Repeat each embedding according to duration
            indices = torch.arange(seq_len, device=x.device)
            repeated_indices = torch.repeat_interleave(indices, dur_b.long())
            
            x_expanded = x_b[repeated_indices]
            output.append(x_expanded)
            output_lengths.append(len(repeated_indices))
        
        # Pad to same length
        max_len = max(output_lengths) if output_lengths else 0
        padded_output = []
        output_mask = []
        
        for b in range(batch_size):
            if b < len(output):
                x_pad = output[b]
                length = output_lengths[b]
                
                if length < max_len:
                    pad_len = max_len - length
                    x_pad = F.pad(x_pad, (0, 0, 0, pad_len))
                
                padded_output.append(x_pad)
                mask = torch.ones(length, device=x.device, dtype=torch.bool)
                if length < max_len:
                    mask = F.pad(mask, (0, pad_len), value=False)
                output_mask.append(mask)
            else:
                padded_output.append(torch.zeros(max_len, embed_dim, device=x.device))
                output_mask.append(torch.zeros(max_len, device=x.device, dtype=torch.bool))
        
        return torch.stack(padded_output), torch.stack(output_mask)


class ConvModule(nn.Module):
    """
    Convolutional module with layer norm and activation.
    
    Args:
        embed_dim: Input/output dimension
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution with norm and activation.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x


class FFTBlock(nn.Module):
    """
    Feed-forward transformer block.
    
    Args:
        config: FastSpeech2Config
        is_decoder: Whether this is a decoder block
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.conv_module = ConvModule(embed_dim, kernel_size, dropout)
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            F.relu,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.dropout(x)
        x = residual + x
        
        # Convolutional module
        residual = x
        x = self.layer_norm2(x)
        x = self.conv_module(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = residual + x
        
        return x


class TextEncoder(nn.Module):
    """
    Text encoder for FastSpeech 2.
    Converts text tokens to contextual embeddings.
    
    Args:
        config: FastSpeech2Config
    """
    
    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.encoder_embed_dim,
            padding_idx=0,
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.encoder_embed_dim,
            max_len=config.max_text_length * 2,
        )
        
        # FFT blocks
        self.layers = nn.ModuleList([
            FFTBlock(
                config.encoder_embed_dim,
                config.encoder_num_heads,
                config.encoder_ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.encoder_num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)
    
    def forward(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Text token IDs (batch, seq_len)
            text_mask: Attention mask for text
            
        Returns:
            Text embeddings (batch, seq_len, embed_dim)
        """
        # Embed tokens
        x = self.embedding(text)
        x = x + self.pos_encoder(x)
        
        # Create causal mask
        seq_len = x.size(1)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        
        # Pass through layers
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=text_mask,
            )
        
        x = self.layer_norm(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder for FastSpeech 2.
    Converts encoder output to mel spectrogram.
    
    Args:
        config: FastSpeech2Config
    """
    
    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        
        self.config = config
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.decoder_embed_dim,
            max_len=config.max_mel_length * 2,
        )
        
        # FFT blocks
        self.layers = nn.ModuleList([
            FFTBlock(
                config.decoder_embed_dim,
                config.decoder_num_heads,
                config.decoder_ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.decoder_num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.decoder_embed_dim)
        
        # Output projection to mel spectrogram
        self.proj = nn.Linear(config.decoder_embed_dim, config.num_mels)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode to mel spectrogram.
        
        Args:
            x: Input embeddings (batch, seq_len, embed_dim)
            attn_mask: Attention mask
            key_padding_mask: Padding mask
            
        Returns:
            Mel spectrogram (batch, seq_len, num_mels)
        """
        # Add positional encoding
        x = x + self.pos_encoder(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        
        # Pass through layers
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )
        
        x = self.layer_norm(x)
        x = self.proj(x)
        
        return x


class SpeakerEmbedding(nn.Module):
    """
    Learnable speaker embedding for voice cloning.
    
    Args:
        num_speakers: Number of unique speakers
        embed_dim: Output embedding dimension
    """
    
    def __init__(self, num_speakers: int, embed_dim: int):
        super().__init__()
        
        self.embedding = nn.Embedding(num_speakers, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, speaker_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(speaker_ids)
        x = self.proj(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]


class FastSpeech2(nn.Module):
    """
    FastSpeech 2 model for voice cloning.
    Provides non-autoregressive TTS with controllable prosody.
    
    Args:
        config: FastSpeech2Config
    """
    
    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        
        self.config = config
        
        # Encoder
        self.encoder = TextEncoder(config)
        
        # Variance predictors
        self.duration_predictor = VariancePredictor(
            config.encoder_embed_dim,
            config.variance_predictor_kernel,
            config.dropout,
        )
        
        self.pitch_predictor = VariancePredictor(
            config.encoder_embed_dim,
            config.variance_predictor_kernel,
            config.dropout,
        )
        
        self.energy_predictor = VariancePredictor(
            config.encoder_embed_dim,
            config.variance_predictor_kernel,
            config.dropout,
        )
        
        # Length regulator
        self.length_regulator = Length Regulator(config.encoder_embed_dim)
        
        # Speaker embedding
        self.speaker_embedding = SpeakerEmbedding(
            config.num_speakers,
            config.speaker_embed_dim,
        )
        self.speaker_proj = nn.Linear(config.speaker_embed_dim, config.encoder_embed_dim)
        
        # Decoder
        self.decoder = Decoder(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        alpha: float = 1.0,  # Duration scaling
        beta: float = 1.0,   # Pitch scaling
        gamma: float = 1.0,  # Energy scaling
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text: Text token IDs (batch, seq_len)
            text_mask: Text attention mask
            duration: Target duration (batch, seq_len)
            pitch: Target pitch (batch, seq_len)
            energy: Target energy (batch, seq_len)
            speaker_ids: Speaker IDs (batch,)
            alpha: Duration scaling factor
            beta: Pitch scaling factor
            gamma: Energy scaling factor
            
        Returns:
            Dictionary containing mel spectrogram and predictions
        """
        # Encode text
        encoder_output = self.encoder(text, text_mask)  # (batch, seq_len, embed_dim)
        
        # Get speaker embedding
        if speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            speaker_emb = self.speaker_proj(speaker_emb)
            # Add speaker embedding
            encoder_output = encoder_output + speaker_emb.unsqueeze(1)
        
        # Predict variance if not provided
        if duration is None:
            duration = self.duration_predictor(encoder_output, text_mask)
        
        if pitch is None:
            pitch = self.pitch_predictor(encoder_output, text_mask)
        
        if energy is None:
            energy = self.energy_predictor(encoder_output, text_mask)
        
        # Scale variance
        duration = (duration * alpha).clamp(min=0)
        pitch = pitch * beta
        energy = energy * gamma
        
        # Length regulation
        expanded_output, output_mask = self.length_regulator(
            encoder_output,
            duration,
            duration_mask=text_mask,
        )
        
        # Decode to mel
        mel = self.decoder(expanded_output, key_padding_mask=~output_mask)
        
        return {
            'mel': mel,
            'output_mask': output_mask,
            'duration_pred': duration,
            'pitch_pred': pitch,
            'energy_pred': energy,
        }
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class HiFiGAN(nn.Module):
    """
    HiFi-GAN vocoder for high-quality audio synthesis.
    Uses multi-period discriminator for natural sound.
    
    Args:
        config: HiFiGANConfig (simplified here)
    """
    
    def __init__(self, config):
        super().__init__()
        # Simplified implementation - full version would include
        # MRD (Multi-Period Discriminator) and MS (Multi-Scale Discriminator)
        self.config = config
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(config.num_mels, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 1, 3, padding=1),
        ])
        
        self.activation = F.leaky_relu
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate audio from mel spectrogram.
        
        Args:
            mel: Mel spectrogram (batch, num_mels, time)
            
        Returns:
            Audio waveform (batch, samples)
        """
        x = mel
        
        for conv in self.conv_blocks:
            x = self.activation(conv(x))
        
        return x.squeeze(1)

