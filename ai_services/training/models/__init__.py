"""
Models module for cinematic music generation.
Provides model architectures for music generation, voice cloning, and lyrics generation.
"""

from .music_transformer import MusicTransformer, MusicTransformerConfig
from .audio_diffusion import AudioDiffusion, AudioDiffusionConfig
from .stem_separator import StemSeparator, StemSeparatorConfig
from .tacotron import Tacotron2, Tacotron2Config
from .fastspeech import FastSpeech2, FastSpeech2Config
from .hifigan import HiFiGAN, HiFiGANConfig
from .lyrics_generator import LyricsGenerator, LyricsGeneratorConfig
from .emotion_classifier import EmotionClassifier, EmotionClassifierConfig

__all__ = [
    'MusicTransformer',
    'MusicTransformerConfig',
    'AudioDiffusion',
    'AudioDiffusionConfig',
    'StemSeparator',
    'StemSeparatorConfig',
    'Tacotron2',
    'Tacotron2Config',
    'FastSpeech2',
    'FastSpeech2Config',
    'HiFiGAN',
    'HiFiGANConfig',
    'LyricsGenerator',
    'LyricsGeneratorConfig',
    'EmotionClassifier',
    'EmotionClassifierConfig',
]
