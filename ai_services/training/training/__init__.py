"""
Training module for cinematic music generation.
Provides training loops, distributed training, and optimization utilities.
"""

from .trainer import BaseTrainer, MusicTrainer, VoiceTrainer, LyricsTrainer
from .distributed import DistributedManager, MixedPrecisionManager
from .optimizers import create_optimizer, create_scheduler
from .ema import ExponentialMovingAverage
from .checkpointing import CheckpointManager

__all__ = [
    'BaseTrainer',
    'MusicTrainer',
    'VoiceTrainer',
    'LyricsTrainer',
    'DistributedManager',
    'MixedPrecisionManager',
    'create_optimizer',
    'create_scheduler',
    'ExponentialMovingAverage',
    'CheckpointManager',
]
