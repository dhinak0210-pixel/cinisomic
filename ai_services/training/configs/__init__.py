"""
Configuration module for cinematic music generation training.
Provides configuration classes and utilities for managing training parameters.
"""

from .model_config import ModelConfig, MusicModelConfig, VoiceModelConfig, LyricsModelConfig
from .training_config import TrainingConfig, OptimizationConfig, LoggingConfig

__all__ = [
    'ModelConfig',
    'MusicModelConfig', 
    'VoiceModelConfig',
    'LyricsModelConfig',
    'TrainingConfig',
    'OptimizationConfig',
    'LoggingConfig',
]
