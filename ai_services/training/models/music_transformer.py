"""
Music Transformer model for cinematic music generation.
Implements a transformer-based architecture for generating musical audio.
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
class MusicTransformerConfig:
    """Configuration for Music Transformer."""
    # Model dimensions
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    ff_dim: int = 3072
    
    # Audio processing
    audio_channels: int = 2
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    num_mels: int = 128
    
    # Training
    dropout: float = 0.1
    max_sequence_length: int = 4096
    
    # Emotion conditioning
    emotion_conditioning: bool = True
    num_emotions: int = 8
    emotion_embed_dim: int = 64
    
    # Generation
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Initialization
    initializer_range: float = 0.02


class RelativePositionBias(nn.Module):
    """
    Relative position bias for transformer.
    Allows the model to learn relative positional relationships.
    """
    
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # Relative position bias table
        self.relative_attention_bias = nn.Embedding(2 * max_distance + 1, num_heads)
        
        # Initialize
        nn.init.normal_(self.relative_attention_bias.weight, mean=0, std=0.02)
    
    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute relative position bias."""
        # Create relative position indices
        device = self.relative_attention_bias.weight.device
        range_vec = torch.arange(key_length, device=device)
        range_matrix = range_vec.unsqueeze(0).expand(query_length, key_length)
        offset = torch.arange(query_length, device=device).unsqueeze(-1)
        
        # Compute relative positions
        relative_positions = range_matrix - offset
        relative_positions = relative_positions.clamp(-self.max_distance, self.max_distance)
        relative_positions = relative_positions + self.max_distance
        
        # Get bias
        bias = self.relative_attention_bias(relative_positions)
        
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, q_len, k_len)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative position bias.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Relative position bias
        self.relative_position_bias = RelativePositionBias(num_heads)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention.
        
        Args:
            query: Query tensor (..., seq_len_q, embed_dim)
            key: Key tensor (..., seq_len_k, embed_dim)
            value: Value tensor (..., seq_len_v, embed_dim)
            key_padding_mask: Padding mask for key (batch, seq_len_k)
            attn_mask: Attention mask
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add relative position bias
        if seq_len_q == seq_len_k:
            relative_bias = self.relative_position_bias(seq_len_q, seq_len_k)
            attn_scores = attn_scores + relative_bias
        
        # Apply masks
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)
        
        # Project output
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Args:
        embed_dim: Embedding dimension
        ff_dim: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """
    Single transformer layer with pre-norm.
    
    Args:
        config: MusicTransformerConfig
    """
    
    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        
        self.feed_forward = FeedForward(
            embed_dim=config.embed_dim,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
        
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm: attention
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask, attn_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm: feed-forward
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class AudioEncoder(nn.Module):
    """
    Encoder for raw audio waveforms.
    Converts audio to latent representations for the transformer.
    
    Args:
        config: MusicTransformerConfig
    """
    
    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Initial convolution to reduce audio to token-like representations
        self.conv = nn.Conv1d(
            config.audio_channels,
            config.embed_dim,
            kernel_size=config.n_fft // 2,
            stride=config.hop_length,
            padding=config.n_fft // 4,
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.embed_dim,
            max_len=config.max_sequence_length,
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to token sequence.
        
        Args:
            audio: Audio waveform (batch, channels, samples)
            
        Returns:
            Encoded sequence (batch, seq_len, embed_dim)
        """
        # Apply convolution
        x = self.conv(audio)
        
        # Remove channel dimension for transformer
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoder(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]


class EmotionEmbedding(nn.Module):
    """
    Learnable embedding for emotion conditioning.
    
    Args:
        num_emotions: Number of emotion classes
        embed_dim: Embedding dimension
    """
    
    def __init__(self, num_emotions: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(emotion_ids)
        x = self.proj(x)
        return x


class MusicTransformer(nn.Module):
    """
    Music Transformer for generating cinematic music.
    Uses relative attention and emotion conditioning.
    
    Args:
        config: MusicTransformerConfig
    """
    
    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Token embeddings (for discretized audio)
        self.token_embedding = nn.Embedding(
            num_embeddings=4096,  # Vocabulary size for audio tokens
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
        )
        
        # Audio encoder for continuous audio
        self.audio_encoder = AudioEncoder(config)
        
        # Emotion conditioning
        if config.emotion_conditioning:
            self.emotion_embedding = EmotionEmbedding(
                num_emotions=config.num_emotions,
                embed_dim=config.emotion_embed_dim,
            )
            self.emotion_proj = nn.Linear(config.emotion_embed_dim, config.embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, 4096)  # Vocabulary size
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper distributions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            audio: Raw audio waveform (batch, channels, samples)
            emotion_ids: Emotion class IDs (batch,)
            attention_mask: Attention mask (batch, seq_len)
            position_ids: Position IDs
            use_cache: Use KV caching for generation
            past_key_values: Cached key-values for generation
            
        Returns:
            Dictionary containing logits and other outputs
        """
        # Determine input type
        if audio is not None:
            # Encode continuous audio
            hidden_states = self.audio_encoder(audio)
        else:
            # Use token embeddings
            hidden_states = self.token_embedding(input_ids)
        
        # Add emotion conditioning
        if emotion_ids is not None and self.config.emotion_conditioning:
            emotion_emb = self.emotion_embedding(emotion_ids)
            emotion_emb = self.emotion_proj(emotion_emb)
            # Add emotion embedding to all positions
            hidden_states = hidden_states + emotion_emb.unsqueeze(1)
        
        # Create attention mask
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # Create causal mask for autoregressive generation
        seq_len = hidden_states.size(1)
        attn_mask = self._create_causal_mask(seq_len, hidden_states.device)
        
        # Pass through transformer layers
        presents = []
        for i, layer in enumerate(self.layers):
            if past_key_values is not None:
                # Use cached key-values
                key, value = past_key_values[i]
                hidden_states = layer(
                    hidden_states,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
                presents.append((key, value))
            else:
                hidden_states = layer(
                    hidden_states,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
        
        # Final layer norm and projection
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        if use_cache and past_key_values is not None:
            output['past_key_values'] = tuple(presents)
        
        return output
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask
    
    def generate(
        self,
        prompt: Optional[torch.Tensor] = None,
        audio_prompt: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Generate music tokens autoregressively.
        
        Args:
            prompt: Starting tokens (batch, seq_len)
            audio_prompt: Starting audio (batch, channels, samples)
            emotion_ids: Emotion class (batch,)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = 1  # Simplified for generation
        
        # Initialize
        if audio_prompt is not None:
            hidden_states = self.audio_encoder(audio_prompt)
            generated = None
        else:
            generated = prompt if prompt is not None else torch.full(
                (batch_size, 1), pad_token_id, dtype=torch.long, device=self.device
            )
            hidden_states = self.token_embedding(generated)
        
        # Cache for generation
        past_key_values = None
        
        # Generation loop
        for _ in range(max_length):
            # Forward pass
            outputs = self.forward(
                input_ids=generated,
                audio=None,
                emotion_ids=emotion_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            logits = outputs['logits'][:, -1, :]  # Get last token logits
            past_key_values = outputs.get('past_key_values')
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k / Top-p sampling
            if do_sample:
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1].unsqueeze(1)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits = logits.scatter(1, indices_to_remove, float('-inf'))
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            if generated is None:
                generated = next_token
            else:
                generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

