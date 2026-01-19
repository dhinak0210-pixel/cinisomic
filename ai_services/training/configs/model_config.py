"""
Model configuration classes for cinematic music generation.
Defines hyperparameters and architecture settings for all model types.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ModelArchitecture(Enum):
    """Supported model architectures."""
    MUSIC_TRANSFORMER = "music_transformer"
    AUDIO_DIFFUSION = "audio_diffusion"
    TACOTRON2 = "tacotron2"
    FASTSpeech2 = "fastspeech2"
    HIFIGAN = "hifigan"
    GPT_NEO = "gpt_neo"
    BART = "bart"
    T5 = "t5"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion-based models."""
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    loss_type: str = "mse"
    schedule_kwargs: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.beta_schedule not in ["linear", "cosine", "sqrt"]:
            raise ValueError(f"Invalid beta_schedule: {self.beta_schedule}")
        if self.loss_type not in ["mse", "l1", "smooth_l1"]:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")


@dataclass
class TransformerConfig:
    """Base transformer configuration."""
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    ff_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    
    
@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    num_channels: int = 2
    audio_length: int = 48000
    segment_duration: float = 10.0
    normalize: bool = True
    remove_silence: bool = True


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = "base_model"
    architecture: ModelArchitecture = ModelArchitecture.MUSIC_TRANSFORMER
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_sequence_length: int = 4096
    gradient_clip_val: float = 1.0
    checkpoint_interval: int = 1000
    keep_last_checkpoints: int = 5
    
    # Pretrained model path
    pretrained_path: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.architecture, str):
            self.architecture = ModelArchitecture(self.architecture)


@dataclass
class MusicModelConfig(ModelConfig):
    """Configuration for music generation models."""
    name: str = "music_transformer"
    
    # Transformer settings
    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig(
        num_layers=12,
        num_heads=12,
        embed_dim=768,
        ff_dim=3072,
        dropout=0.1
    ))
    
    # Audio settings
    audio: AudioConfig = field(default_factory=lambda: AudioConfig(
        sample_rate=44100,
        num_channels=2,
        audio_length=48000
    ))
    
    # Latent dimension for audio
    latent_dim: int = 512
    
    # Diffusion settings (optional)
    diffusion: Optional[DiffusionConfig] = None
    
    # Stem separation settings
    num_stems: int = 5  # e.g., drums, bass, vocals, other, melody
    
    # Emotion conditioning
    emotion_conditioning: bool = True
    num_emotions: int = 8
    
    def __post_init__(self):
        super().__post_init__()
        self.architecture = ModelArchitecture.MUSIC_TRANSFORMER
        
        if self.diffusion is not None and isinstance(self.diffusion, dict):
            self.diffusion = DiffusionConfig(**self.diffusion)


@dataclass
class VoiceModelConfig(ModelConfig):
    """Configuration for voice cloning models."""
    name: str = "fastspeech2"
    
    # Encoder configuration
    encoder: TransformerConfig = field(default_factory=lambda: TransformerConfig(
        num_layers=4,
        num_heads=4,
        embed_dim=256,
        ff_dim=1024,
        dropout=0.1
    ))
    
    # Decoder configuration
    decoder: TransformerConfig = field(default_factory=lambda: TransformerConfig(
        num_layers=4,
        num_heads=4,
        embed_dim=256,
        ff_dim=1024,
        dropout=0.1
    ))
    
    # Text processing
    max_text_length: int = 200
    vocab_size: int = 300
    
    # Audio processing
    audio: AudioConfig = field(default_factory=lambda: AudioConfig(
        sample_rate=22050,
        num_channels=1,
        audio_length=22050
    ))
    
    # Mel spectrogram settings
    num_mels: int = 80
    mel_length: int = 1024
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    
    # Speaker embedding
    num_speakers: int = 100
    speaker_embed_dim: int = 192
    
    # Vocoder
    vocoder: str = "hifigan"
    vocoder_config: Optional[Dict] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.architecture = ModelArchitecture.FASTSpeech2


@dataclass
class LyricsModelConfig(ModelConfig):
    """Configuration for lyrics generation models."""
    name: str = "gpt_neo"
    
    # Model size settings
    model_size: str = "large"  # small, medium, large
    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig(
        num_layers=12,
        num_heads=12,
        embed_dim=768,
        ff_dim=3072,
        dropout=0.1
    ))
    
    # Text processing
    max_length: int = 512
    vocab_size: int = 50257
    
    # Multilingual support
    languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "it", "ja", "zh"
    ])
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
    
    def __post_init__(self):
        super().__post_init__()
        self.architecture = ModelArchitecture.GPT_NEO
        
        # Adjust transformer size based on model_size
        size_configs = {
            "small": (6, 8, 512, 2048),
            "medium": (8, 16, 1024, 4096),
            "large": (12, 16, 1536, 6144)
        }
        
        if self.model_size in size_configs:
            layers, heads, dim, ff_dim = size_configs[self.model_size]
            self.transformer.num_layers = layers
            self.transformer.num_heads = heads
            self.transformer.embed_dim = dim
            self.transformer.ff_dim = ff_dim


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: List[str] = field(default_factory=lambda: [
        "fid", "clap_score", "mos"
    ])
    eval_batch_size: int = 16
    num_samples: int = 100
    generation_length: int = 30  # seconds
    
    # Subjective evaluation
    subjective_evaluation: bool = True
    num_raters: int = 10
    questions: List[str] = field(default_factory=lambda: [
        "overall_quality", "relevance", "creativity", "technical_quality"
    ])
