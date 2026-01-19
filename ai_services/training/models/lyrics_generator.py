"""
Lyrics Generator for multilingual lyrics generation.
Implements GPT-based model with emotion and scene context conditioning.
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
class LyricsGeneratorConfig:
    """Configuration for Lyrics Generator."""
    # Model dimensions
    vocab_size: int = 50257
    max_length: int = 512
    
    # Transformer dimensions
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    ff_dim: int = 3072
    
    # Training
    dropout: float = 0.1
    
    # Multilingual support
    languages: List[str] = None
    default_language: str = "en"
    
    # Emotion conditioning
    emotion_conditioning: bool = True
    num_emotions: int = 8
    emotion_embed_dim: int = 64
    
    # Genre conditioning
    genre_conditioning: bool = True
    num_genres: int = 20
    genre_embed_dim: int = 64
    
    # Scene context
    scene_context: bool = True
    scene_embed_dim: int = 128
    
    # Initialization
    initializer_range: float = 0.02


class MultiLingualEmbedding(nn.Module):
    """
    Language-specific token embeddings.
    Allows for better handling of multiple languages.
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_languages: Number of languages
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_languages: int = 10,
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.language_embedding = nn.Embedding(num_languages, embed_dim)
        
        self.num_languages = num_languages
    
    def forward(
        self,
        input_ids: torch.Tensor,
        language_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get embeddings for tokens and language.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            language_ids: Language IDs (batch,)
            
        Returns:
            Combined embeddings (batch, seq_len, embed_dim)
        """
        token_emb = self.token_embedding(input_ids)
        
        if language_ids is not None:
            lang_emb = self.language_embedding(language_ids)
            # Add language embedding to all positions
            token_emb = token_emb + lang_emb.unsqueeze(1)
        
        return token_emb


class EmotionEmbedding(nn.Module):
    """
    Emotion conditioning embedding.
    
    Args:
        num_emotions: Number of emotion classes
        embed_dim: Output embedding dimension
    """
    
    def __init__(self, num_emotions: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(emotion_ids)
        x = self.proj(x)
        return x


class GenreEmbedding(nn.Module):
    """
    Genre conditioning embedding.
    
    Args:
        num_genres: Number of genre classes
        embed_dim: Output embedding dimension
    """
    
    def __init__(self, num_genres: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_genres, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, genre_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(genre_ids)
        x = self.proj(x)
        return x


class SceneContextEmbedding(nn.Module):
    """
    Scene context embedding for conditional generation.
    
    Args:
        scene_embed_dim: Scene embedding dimension
        embed_dim: Output embedding dimension
    """
    
    def __init__(self, scene_embed_dim: int, embed_dim: int):
        super().__init__()
        
        # Project scene features
        self.scene_proj = nn.Linear(scene_embed_dim, embed_dim)
        
        # Learnable position embedding for scene context
        self.scene_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
    
    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """
        Project scene features to embeddings.
        
        Args:
            scene_features: Scene context features (batch, scene_embed_dim)
            
        Returns:
            Scene embedding (batch, embed_dim)
        """
        x = self.scene_proj(scene_features)
        x = x + self.scene_pos_embed
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with GELU activation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = residual + x
        
        # Feed-forward with pre-norm
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
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


class LyricsGenerator(nn.Module):
    """
    GPT-based lyrics generator with multilingual support.
    Conditions on emotion, genre, and scene context.
    
    Args:
        config: LyricsGeneratorConfig
    """
    
    def __init__(self, config: LyricsGeneratorConfig):
        super().__init__()
        
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Language embedding
        self.language_embedding = nn.Embedding(
            len(config.languages) if config.languages else 10,
            config.embed_dim,
        )
        
        # Language mapping
        self.language_map = {lang: i for i, lang in enumerate(config.languages)} if config.languages else {}
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.embed_dim,
            max_len=config.max_length,
        )
        
        # Emotion conditioning
        if config.emotion_conditioning:
            self.emotion_embedding = EmotionEmbedding(
                config.num_emotions,
                config.embed_dim,
            )
        
        # Genre conditioning
        if config.genre_conditioning:
            self.genre_embedding = GenreEmbedding(
                config.num_genres,
                config.embed_dim,
            )
        
        # Scene context
        if config.scene_context:
            self.scene_embedding = SceneContextEmbedding(
                config.scene_embed_dim,
                config.embed_dim,
            )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ff_dim,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        genre_ids: Optional[torch.Tensor] = None,
        scene_features: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            language_ids: Language IDs (batch,)
            emotion_ids: Emotion class IDs (batch,)
            genre_ids: Genre class IDs (batch,)
            scene_features: Scene context features (batch, scene_embed_dim)
            position_ids: Position IDs
            use_cache: Use KV caching
            past_key_values: Cached KV pairs
            
        Returns:
            Dictionary with logits and other outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add language embedding
        if language_ids is not None:
            lang_emb = self.language_embedding(language_ids)
            hidden_states = hidden_states + lang_emb.unsqueeze(1)
        
        # Add positional encoding
        hidden_states = hidden_states + self.pos_encoder(hidden_states)
        
        # Add emotion conditioning
        if emotion_ids is not None and self.config.emotion_conditioning:
            emotion_emb = self.emotion_embedding(emotion_ids)
            hidden_states = hidden_states + emotion_emb.unsqueeze(1)
        
        # Add genre conditioning
        if genre_ids is not None and self.config.genre_conditioning:
            genre_emb = self.genre_embedding(genre_ids)
            hidden_states = hidden_states + genre_emb.unsqueeze(1)
        
        # Add scene context
        if scene_features is not None and self.config.scene_context:
            scene_emb = self.scene_embedding(scene_features)
            # Add scene embedding to first position
            hidden_states[:, 0] = hidden_states[:, 0] + scene_emb
        
        # Create masks
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        
        # Pass through transformer layers
        presents = []
        for i, layer in enumerate(self.layers):
            if past_key_values is not None:
                key, value = past_key_values[i]
                hidden_states = layer(
                    hidden_states,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask,
                )
                presents.append((key, value))
            else:
                hidden_states = layer(
                    hidden_states,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask,
                )
        
        # Final norm and projection
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        if use_cache and past_key_values is not None:
            output['past_key_values'] = tuple(presents)
        
        return output
    
    def generate(
        self,
        prompt: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        genre_ids: Optional[torch.Tensor] = None,
        scene_features: Optional[torch.Tensor] = None,
        max_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Generate lyrics autoregressively.
        
        Args:
            prompt: Starting tokens (batch, seq_len)
            language_ids: Language IDs (batch,)
            emotion_ids: Emotion class IDs (batch,)
            genre_ids: Genre class IDs (batch,)
            scene_features: Scene context (batch, scene_embed_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to sample
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs
        """
        batch_size = 1
        
        # Initialize
        if prompt is not None:
            generated = prompt
        else:
            generated = torch.full(
                (batch_size, 1), pad_token_id, dtype=torch.long, device=self.device
            )
        
        past_key_values = None
        
        for _ in range(max_length):
            outputs = self.forward(
                input_ids=generated,
                language_ids=language_ids,
                emotion_ids=emotion_ids,
                genre_ids=genre_ids,
                scene_features=scene_features,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']
            
            # Temperature
            logits = logits / temperature
            
            # Top-k / Top-p
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
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_language_id(self, language: str) -> torch.Tensor:
        """
        Get language ID tensor for a language string.
        
        Args:
            language: Language string (e.g., 'en', 'es')
            
        Returns:
            Language ID tensor (1,)
        """
        lang_id = self.language_map.get(language, 0)
        return torch.tensor([lang_id], device=self.device)

