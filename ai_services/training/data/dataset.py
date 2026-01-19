"""
Base dataset classes for cinematic music generation.
Provides abstract base classes and common functionality for all dataset types.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base dataset class for all cinematic music datasets.
    Provides common functionality for audio loading, caching, and transformations.
    
    Args:
        data_root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to apply to samples
        cache: Whether to cache samples in memory
        max_samples: Maximum number of samples to load (None for all)
    """
    
    SUPPORTED_FORMATS = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
        **kwargs
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.cache = cache
        self.cache_dict = {}
        self.max_samples = max_samples
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate split
        if split not in ['train', 'val', 'test', 'all']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test', 'all']")
        
        # Initialize dataset
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """
        Load and return list of sample metadata.
        Override in subclasses.
        
        Returns:
            List of sample dictionaries containing file paths and metadata
        """
        raise NotImplementedError("Subclasses must implement _load_samples")
    
    def _load_audio(self, filepath: str) -> np.ndarray:
        """
        Load audio file from disk.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            import librosa
            audio, sr = librosa.load(filepath, sr=None, mono=False)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio {filepath}: {e}")
            raise
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Normalized audio waveform
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _remove_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Remove silence from beginning and end of audio.
        
        Args:
            audio: Input audio waveform
            threshold_db: Silence threshold in dB
            
        Returns:
            Audio with silence removed
        """
        try:
            import librosa
            # Convert threshold to amplitude
            threshold = 10 ** (threshold_db / 20)
            
            # Find non-silent intervals
            intervals = librosa.effects.split(audio, top_db=-threshold_db)
            
            if len(intervals) > 0:
                # Concatenate non-silent segments
                audio = np.concatenate([audio[start:end] for start, end in intervals])
            
            return audio
        except Exception as e:
            logger.warning(f"Error removing silence: {e}")
            return audio
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary containing audio, metadata, and transformations
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        # Check cache
        if self.cache and idx in self.cache_dict:
            return self.cache_dict[idx]
        
        # Get sample metadata
        sample = self.samples[idx].copy()
        
        # Load audio if needed
        if 'audio' not in sample and 'filepath' in sample:
            audio = self._load_audio(sample['filepath'])
            
            # Apply preprocessing
            audio = self._normalize_audio(audio)
            audio = self._remove_silence(audio)
            
            sample['audio'] = audio
        
        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Cache if enabled
        if self.cache:
            self.cache_dict[idx] = sample
        
        return sample
    
    def get_sample_metadata(self, idx: int) -> Dict:
        """
        Get metadata for a sample without loading audio.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample metadata dictionary
        """
        return self.samples[idx].copy()
    
    def filter_by_duration(
        self, 
        min_duration: float = 0.0, 
        max_duration: float = float('inf')
    ) -> 'BaseDataset':
        """
        Filter samples by duration.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Filtered dataset
        """
        filtered_samples = [
            sample for sample in self.samples
            if min_duration <= sample.get('duration', 0) <= max_duration
        ]
        
        self.samples = filtered_samples
        logger.info(f"Filtered to {len(self.samples)} samples")
        return self
    
    def filter_by_metadata(
        self, 
        metadata_filter: Dict[str, Any]
    ) -> 'BaseDataset':
        """
        Filter samples by metadata criteria.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            
        Returns:
            Filtered dataset
        """
        filtered_samples = []
        for sample in self.samples:
            match = True
            for key, value in metadata_filter.items():
                if sample.get(key) != value:
                    match = False
                    break
            if match:
                filtered_samples.append(sample)
        
        self.samples = filtered_samples
        logger.info(f"Filtered to {len(self.samples)} samples")
        return self
    
    def sample(self, n: int) -> List[Dict]:
        """
        Randomly sample n samples from the dataset.
        
        Args:
            n: Number of samples to sample
            
        Returns:
            List of sampled dictionaries
        """
        return random.sample(self.samples, min(n, len(self.samples)))
    
    def get_class_distribution(self, class_key: str) -> Dict[str, int]:
        """
        Get the distribution of classes for a given metadata key.
        
        Args:
            class_key: Metadata key to get distribution for
            
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {}
        for sample in self.samples:
            class_name = str(sample.get(class_key, 'unknown'))
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)}, split='{self.split}')"


class MusicDataset(BaseDataset):
    """
    Dataset for music audio files.
    Handles loading and preprocessing of cinematic music tracks.
    
    Args:
        data_root: Root directory containing music files
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to apply
        cache: Whether to cache samples
        max_samples: Maximum samples to load
        segment_duration: Duration of audio segments in seconds
        sample_rate: Target sample rate
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache: bool = False,
        max_samples: Optional[int] = None,
        segment_duration: float = 10.0,
        sample_rate: int = 44100,
        seed: int = 42,
        **kwargs
    ):
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            cache=cache,
            max_samples=max_samples,
            seed=seed,
            **kwargs
        )
    
    def _load_samples(self) -> List[Dict]:
        """
        Load music samples from the dataset directory.
        
        Returns:
            List of sample dictionaries with file paths and metadata
        """
        samples = []
        
        # Search for audio files
        for ext in self.SUPPORTED_FORMATS:
            audio_files = list(self.data_root.rglob(f"*{ext}"))
            
            for filepath in audio_files:
                # Get relative path for sample ID
                rel_path = filepath.relative_to(self.data_root)
                
                # Try to load metadata from adjacent files
                metadata = self._load_metadata(filepath)
                
                sample = {
                    'id': str(rel_path).replace('/', '_').replace(ext, ''),
                    'filepath': str(filepath),
                    'duration': metadata.get('duration', 0.0),
                    'tempo': metadata.get('tempo', 120.0),
                    'key': metadata.get('key', 'C'),
                    'emotion': metadata.get('emotion', 'neutral'),
                    'genre': metadata.get('genre', 'cinematic'),
                    'source': str(rel_path.parent),
                    'split': self.split,
                    'metadata': metadata
                }
                
                samples.append(sample)
                
                if self.max_samples and len(samples) >= self.max_samples:
                    break
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        # Shuffle if training split
        if self.split == 'train':
            random.shuffle(samples)
        
        return samples
    
    def _load_metadata(self, filepath: Path) -> Dict:
        """
        Load metadata for an audio file.
        Looks for adjacent JSON or CSV files.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Try adjacent JSON file
        json_path = filepath.with_suffix('.json')
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        
        # Try adjacent CSV file
        csv_path = filepath.with_suffix('.csv')
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                metadata = df.iloc[0].to_dict()
        
        return metadata
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a music sample with potential segmentation.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary with audio tensor and metadata
        """
        sample = super().__getitem__(idx)
        
        if 'audio' in sample:
            audio = sample['audio']
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Resample if necessary
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Segment if audio is longer than segment_duration
            if audio_tensor.shape[1] > self.segment_samples:
                start = random.randint(0, audio_tensor.shape[1] - self.segment_samples)
                audio_tensor = audio_tensor[:, start:start + self.segment_samples]
            
            sample['audio'] = audio_tensor
            sample['segment_length'] = self.segment_samples
            sample['sample_rate'] = self.sample_rate
        
        return sample


class VoiceDataset(BaseDataset):
    """
    Dataset for voice cloning training.
    Handles loading voice samples with speaker embeddings and transcripts.
    
    Args:
        data_root: Root directory containing voice data
        split: Dataset split
        transform: Optional transform to apply
        cache: Whether to cache samples
        max_samples: Maximum samples to load
        speaker_ids: Optional list of speaker IDs to include
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache: bool = False,
        max_samples: Optional[int] = None,
        speaker_ids: Optional[List[str]] = None,
        seed: int = 42,
        **kwargs
    ):
        self.speaker_ids = speaker_ids
        
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            cache=cache,
            max_samples=max_samples,
            seed=seed,
            **kwargs
        )
    
    def _load_samples(self) -> List[Dict]:
        """
        Load voice samples from the dataset.
        
        Returns:
            List of sample dictionaries with audio, text, and speaker info
        """
        samples = []
        
        # Load from speaker directories
        if self.data_root.is_dir():
            for speaker_dir in self.data_root.iterdir():
                if not speaker_dir.is_dir():
                    continue
                
                # Filter by speaker IDs if specified
                if self.speaker_ids and speaker_dir.name not in self.speaker_ids:
                    continue
                
                speaker_id = speaker_dir.name
                
                # Load samples for this speaker
                for ext in self.SUPPORTED_FORMATS:
                    audio_files = list(speaker_dir.rglob(f"*{ext}"))
                    
                    for filepath in audio_files:
                        rel_path = filepath.relative_to(self.data_root)
                        
                        # Load transcript
                        transcript = self._load_transcript(filepath)
                        
                        sample = {
                            'id': str(rel_path).replace('/', '_').replace(ext, ''),
                            'filepath': str(filepath),
                            'speaker_id': speaker_id,
                            'text': transcript.get('text', ''),
                            'text_tokens': transcript.get('tokens', []),
                            'duration': transcript.get('duration', 0.0),
                            'emotion': transcript.get('emotion', 'neutral'),
                            'split': self.split
                        }
                        
                        samples.append(sample)
                        
                        if self.max_samples and len(samples) >= self.max_samples:
                            break
                    
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
        
        # Shuffle if training
        if self.split == 'train':
            random.shuffle(samples)
        
        return samples
    
    def _load_transcript(self, filepath: Path) -> Dict:
        """
        Load transcript for an audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Transcript dictionary
        """
        transcript = {}
        
        # Try adjacent text file
        txt_path = filepath.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                transcript['text'] = f.read().strip()
        
        # Try adjacent JSON file for additional metadata
        json_path = filepath.with_suffix('.json')
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                transcript.update(json.load(f))
        
        return transcript
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a voice sample with text and speaker info.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary with audio tensor, text, and speaker ID
        """
        sample = super().__getitem__(idx)
        
        if 'audio' in sample:
            audio = sample['audio']
            audio_tensor = torch.from_numpy(audio).float()
            
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            sample['audio'] = audio_tensor
        
        return sample


class LyricsDataset(BaseDataset):
    """
    Dataset for lyrics generation training.
    Handles loading lyrics with emotion and scene context labels.
    
    Args:
        data_root: Root directory containing lyrics data
        split: Dataset split
        transform: Optional transform to apply
        cache: Whether to cache samples
        max_samples: Maximum samples to load
        languages: Optional list of languages to include
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache: bool = False,
        max_samples: Optional[int] = None,
        languages: Optional[List[str]] = None,
        seed: int = 42,
        **kwargs
    ):
        self.languages = languages
        
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            cache=cache,
            max_samples=max_samples,
            seed=seed,
            **kwargs
        )
    
    def _load_samples(self) -> List[Dict]:
        """
        Load lyrics samples from the dataset.
        
        Returns:
            List of sample dictionaries with lyrics and metadata
        """
        samples = []
        
        # Load from text files and metadata
        if self.data_root.is_dir():
            for filepath in self.data_root.rglob('*.txt'):
                rel_path = filepath.relative_to(self.data_root)
                
                # Check language filter
                if self.languages:
                    lang = rel_path.parts[0] if len(rel_path.parts) > 0 else 'unknown'
                    if lang not in self.languages:
                        continue
                
                # Load lyrics
                with open(filepath, 'r', encoding='utf-8') as f:
                    lyrics = f.read().strip()
                
                # Load metadata
                metadata = self._load_lyrics_metadata(filepath)
                
                sample = {
                    'id': str(rel_path).replace('/', '_').replace('.txt', ''),
                    'filepath': str(filepath),
                    'lyrics': lyrics,
                    'language': metadata.get('language', 'en'),
                    'emotion': metadata.get('emotion', 'neutral'),
                    'genre': metadata.get('genre', 'cinematic'),
                    'scene_context': metadata.get('scene_context', ''),
                    'tempo': metadata.get('tempo', 120),
                    'mood': metadata.get('mood', 'neutral'),
                    'split': self.split
                }
                
                samples.append(sample)
                
                if self.max_samples and len(samples) >= self.max_samples:
                    break
        
        # Shuffle if training
        if self.split == 'train':
            random.shuffle(samples)
        
        return samples
    
    def _load_lyrics_metadata(self, filepath: Path) -> Dict:
        """
        Load metadata for lyrics.
        
        Args:
            filepath: Path to lyrics file
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Try adjacent JSON file
        json_path = filepath.with_suffix('.json')
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        
        # Try parent directory metadata
        parent_meta = filepath.parent / 'metadata.json'
        if parent_meta.exists():
            import json
            with open(parent_meta, 'r') as f:
                parent_data = json.load(f)
                metadata.update(parent_data)
        
        return metadata
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a lyrics sample with all metadata.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary with lyrics text and labels
        """
        return super().__getitem__(idx)


class MetadataDataset(BaseDataset):
    """
    Dataset for handling metadata without audio.
    Useful for metadata-only operations and analysis.
    """
    
    def __init__(
        self,
        metadata_path: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
        **kwargs
    ):
        self.metadata_path = Path(metadata_path)
        
        super().__init__(
            data_root=str(metadata_path),
            split=split,
            transform=transform,
            cache=cache,
            max_samples=max_samples,
            seed=seed,
            **kwargs
        )
    
    def _load_samples(self) -> List[Dict]:
        """
        Load samples from metadata CSV or JSON file.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        if self.metadata_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(self.metadata_path)
            
            for idx, row in df.iterrows():
                sample = row.to_dict()
                sample['id'] = str(idx)
                samples.append(sample)
                
                if self.max_samples and len(samples) >= self.max_samples:
                    break
        
        elif self.metadata_path.suffix == '.json':
            import json
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    item['id'] = item.get('id', str(idx))
                    samples.append(item)
                    
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
            elif isinstance(data, dict):
                samples.append(data)
        
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a metadata sample."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.samples[idx].copy()
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

