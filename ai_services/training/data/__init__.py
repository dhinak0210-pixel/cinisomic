"""
Data module for cinematic music generation training.
Provides data loading, preprocessing, and augmentation utilities.
"""

from .dataset import BaseDataset, MusicDataset, VoiceDataset, LyricsDataset, MetadataDataset
from .preprocessing import AudioPreprocessor, FeatureExtractor
from .augmentation import AudioAugmentor, DataAugmentationPipeline
from .cleaning import DataCleaner, BiasReducer
from .copyright import CopyrightSafeLoader
from .loaders import create_data_loaders, DistributedSampler

__all__ = [
    'BaseDataset',
    'MusicDataset',
    'VoiceDataset', 
    'LyricsDataset',
    'MetadataDataset',
    'AudioPreprocessor',
    'FeatureExtractor',
    'AudioAugmentor',
    'DataAugmentationPipeline',
    'DataCleaner',
    'BiasReducer',
    'CopyrightSafeLoader',
    'create_data_loaders',
    'DistributedSampler',
]
