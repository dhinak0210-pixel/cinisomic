"""
Data augmentation module for cinematic music generation.
Provides extensive audio augmentation techniques to generate 1000+ variations.
"""

import numpy as np
import torch
import random
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    # Tempo variations
    tempo_enabled: bool = True
    tempo_range: Tuple[float, float] = (0.85, 1.15)  # ±15%
    
    # Pitch variations
    pitch_enabled: bool = True
    pitch_range: Tuple[float, float] = (-3.0, 3.0)  # ±3 semitones
    
    # Dynamic range
    dynamic_enabled: bool = True
    compression_ratio: float = 4.0
    
    # Reverb
    reverb_enabled: bool = True
    reverb_room_size_range: Tuple[float, float] = (0.2, 0.8)
    
    # Noise injection
    noise_enabled: bool = True
    noise_level_range: Tuple[float, float] = (0.001, 0.02)
    noise_types: List[str] = field(default_factory=lambda: ['white', 'pink', 'brown'])
    
    # EQ variations
    eq_enabled: bool = True
    eq_boost_range: Tuple[float, float] = (-6.0, 6.0)
    
    # Stereo width
    stereo_enabled: bool = True
    stereo_width_range: Tuple[float, float] = (0.5, 1.5)
    
    # Time shift
    time_shift_enabled: bool = True
    time_shift_range: Tuple[float, float] = (-0.1, 0.1)  # ±10%
    
    # Volume
    volume_enabled: bool = True
    volume_range: Tuple[float, float] = (0.7, 1.0)
    
    # Number of variations to generate
    variation_count: int = 1000


class AugmentationPipeline:
    """
    Comprehensive audio augmentation pipeline.
    Generates diverse variations for robust model training.
    
    Args:
        config: Augmentation configuration
        sample_rate: Audio sample rate
        random_seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        sample_rate: int = 44100,
        random_seed: int = 42,
    ):
        self.config = config or AugmentationConfig()
        self.sample_rate = sample_rate
        self.random_seed = random_seed
        
        # Initialize random state
        self.reset_seed()
        
        # Initialize augmentations
        self._setup_augmentations()
    
    def reset_seed(self):
        """Reset random seed for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
    
    def _setup_augmentations(self):
        """Setup augmentation functions."""
        self.augmentations = {}
        
        if self.config.tempo_enabled:
            self.augmentations['tempo'] = self._apply_tempo
            
        if self.config.pitch_enabled:
            self.augmentations['pitch'] = self._apply_pitch
            
        if self.config.dynamic_enabled:
            self.augmentations['dynamic'] = self._apply_dynamic
            
        if self.config.reverb_enabled:
            self.augmentations['reverb'] = self._apply_reverb
            
        if self.config.noise_enabled:
            self.augmentations['noise'] = self._apply_noise
            
        if self.config.eq_enabled:
            self.augmentations['eq'] = self._apply_eq
            
        if self.config.stereo_enabled:
            self.augmentations['stereo'] = self._apply_stereo
            
        if self.config.time_shift_enabled:
            self.augmentations['time_shift'] = self._apply_time_shift
            
        if self.config.volume_enabled:
            self.augmentations['volume'] = self._apply_volume
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Augmented audio waveform
        """
        augmented = audio.copy()
        
        # Shuffle augmentation order for variety
        aug_names = list(self.augmentations.keys())
        random.shuffle(aug_names)
        
        # Apply each augmentation with random parameters
        for name in aug_names:
            if random.random() < 0.7:  # 70% chance to apply each augmentation
                augmented = self.augmentations[name](augmented)
        
        return augmented
    
    def _apply_tempo(self, audio: np.ndarray) -> np.ndarray:
        """Apply tempo variation."""
        try:
            import librosa
            rate = random.uniform(*self.config.tempo_range)
            if len(audio.shape) > 1:
                stretched = []
                for channel in audio:
                    stretched.append(librosa.effects.time_stretch(channel, rate=rate))
                return np.array(stretched)
            else:
                return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            logger.warning(f"Tempo augmentation failed: {e}")
            return audio
    
    def _apply_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shift."""
        try:
            import librosa
            n_steps = random.uniform(*self.config.pitch_range)
            if len(audio.shape) > 1:
                shifted = []
                for channel in audio:
                    shifted.append(librosa.effects.pitch_shift(channel, sr=self.sample_rate, n_steps=n_steps))
                return np.array(shifted)
            else:
                return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        except Exception as e:
            logger.warning(f"Pitch augmentation failed: {e}")
            return audio
    
    def _apply_dynamic(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        threshold = random.uniform(-30, -10)  # dB
        ratio = self.config.compression_ratio
        
        # Simple soft compression
        threshold_val = 10 ** (threshold / 20)
        compressed = audio.copy()
        
        above_threshold = np.abs(compressed) > threshold_val
        if np.any(above_threshold):
            overshoot = (np.abs(compressed[above_threshold]) - threshold_val) / np.abs(compressed[above_threshold])
            gain = 1 - (overshoot * (1 - 1/ratio))
            compressed[above_threshold] = compressed[above_threshold] * np.clip(gain, 0.3, 1.0)
        
        return compressed
    
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply reverb effect."""
        try:
            from scipy.signal import convolve
            
            room_size = random.uniform(*self.config.reverb_room_size_range)
            wet_level = random.uniform(0.1, 0.4)
            dry_level = 1.0 - wet_level * 0.5
            
            # Create impulse response
            length = int(self.sample_rate * room_size * 2)
            decay = np.exp(-np.linspace(0, 8, length))
            ir = decay * (0.9 + np.random.randn(length) * 0.1)
            ir = ir / np.max(np.abs(ir)) if np.max(np.abs(ir)) > 0 else ir
            
            if len(audio.shape) > 1:
                reverb = []
                for channel in audio:
                    reverb.append(convolve(channel, ir, mode='same'))
                reverb = np.array(reverb)
            else:
                reverb = convolve(audio, ir, mode='same')
            
            return dry_level * audio + wet_level * reverb
            
        except Exception as e:
            logger.warning(f"Reverb augmentation failed: {e}")
            return audio
    
    def _apply_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise injection."""
        noise_type = random.choice(self.config.noise_types)
        noise_level = random.uniform(*self.config.noise_level_range)
        
        # Calculate signal level
        signal_level = np.max(np.abs(audio))
        
        # Generate noise
        if noise_type == 'white':
            noise = np.random.randn(*audio.shape).astype(np.float32)
        elif noise_type == 'pink':
            white = np.random.randn(*audio.shape).astype(np.float32)
            noise = np.cumsum(white) / np.sqrt(np.arange(1, audio.shape[0] + 1))
            if len(audio.shape) > 1:
                noise = noise / np.max(np.abs(noise)) if np.max(np.abs(noise)) > 0 else noise
            else:
                noise = noise / np.max(np.abs(noise)) if np.max(np.abs(noise)) > 0 else noise
        elif noise_type == 'brown':
            white = np.random.randn(*audio.shape).astype(np.float32)
            noise = np.cumsum(white)
            noise = noise / np.max(np.abs(noise)) if np.max(np.abs(noise)) > 0 else noise
        else:
            noise = np.random.randn(*audio.shape).astype(np.float32)
        
        # Scale and apply
        if signal_level > 0:
            noise = noise * signal_level * noise_level
        else:
            noise = noise * noise_level
        
        return audio + noise
    
    def _apply_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple EQ variations."""
        try:
            from scipy.signal import butter, lfilter
            
            # Create random EQ bands
            bands = 4
            gains = []
            for _ in range(bands):
                gain = random.uniform(*self.config.eq_boost_range)
                gains.append(gain)
            
            # Apply simple bass/treble adjustment
            # Bass boost/cut
            bass_gain = 1 + gains[0] / 20
            audio = audio * bass_gain
            
            # Simple high-frequency adjustment
            treble_gain = 1 + gains[-1] / 20
            
            # Apply gentle treble boost using simple filtering
            nyquist = self.sample_rate / 2
            low_freq = 2000 / nyquist
            b, a = butter(2, low_freq, btype='high')
            treble = lfilter(b, a, audio)
            
            if len(audio.shape) > 1:
                audio = audio + (treble * (treble_gain - 1) * 0.3)
            else:
                audio = audio + (treble * (treble_gain - 1) * 0.3)
            
            return audio
            
        except Exception as e:
            logger.warning(f"EQ augmentation failed: {e}")
            return audio
    
    def _apply_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Apply stereo width variation."""
        if len(audio.shape) == 1:
            return audio
        
        width = random.uniform(*self.config.stereo_width_range)
        
        left, right = audio[0], audio[1]
        
        # Mix to mono and back
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Adjust stereo width
        side = side * width
        
        # Recombine
        left = mid + side
        right = mid - side
        
        # Normalize
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left = left / max_val * 0.95
            right = right / max_val * 0.95
        
        return np.array([left, right])
    
    def _apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply small time shift."""
        shift = random.uniform(*self.config.time_shift_range)
        shift_samples = int(len(audio) * shift) if len(audio.shape) == 1 else int(audio.shape[1] * shift)
        
        if abs(shift_samples) < 1:
            return audio
        
        if shift_samples > 0:
            # Shift right, pad left
            if len(audio.shape) == 1:
                return np.pad(audio[:-shift_samples], (shift_samples, 0), mode='reflect')
            else:
                padded = np.pad(audio[:, :-shift_samples], ((0, 0), (shift_samples, 0)), mode='reflect')
                return padded[:, :audio.shape[1]]
        else:
            # Shift left, pad right
            shift_samples = abs(shift_samples)
            if len(audio.shape) == 1:
                return np.pad(audio[-shift_samples:], (0, shift_samples), mode='reflect')
            else:
                padded = np.pad(audio[:, -shift_samples:], ((0, 0), (0, shift_samples)), mode='reflect')
                return padded[:, :audio.shape[1]]
    
    def _apply_volume(self, audio: np.ndarray) -> np.ndarray:
        """Apply volume variation."""
        volume = random.uniform(*self.config.volume_range)
        return audio * volume
    
    def generate_variations(
        self,
        audio: np.ndarray,
        num_variations: int,
        return_metadata: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Dict]]]:
        """
        Generate multiple variations of an audio sample.
        
        Args:
            audio: Input audio
            num_variations: Number of variations to generate
            return_metadata: Whether to return augmentation metadata
            
        Returns:
            List of variations, optionally with metadata
        """
        variations = []
        metadata_list = []
        
        for i in range(num_variations):
            # Set different seed for each variation
            seed = self.random_seed + i
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate variation
            var_audio = self(audio)
            variations.append(var_audio)
            
            if return_metadata:
                metadata = {
                    'variation_id': i,
                    'seed': seed,
                    'tempo_rate': random.uniform(*self.config.tempo_range) if self.config.tempo_enabled else None,
                    'pitch_shift': random.uniform(*self.config.pitch_range) if self.config.pitch_enabled else None,
                }
                metadata_list.append(metadata)
        
        # Reset seed
        self.reset_seed()
        
        if return_metadata:
            return variations, metadata_list
        return variations


class StemAugmentor:
    """
    Augmentor specifically designed for multi-stem audio.
    Applies consistent augmentations across all stems.
    
    Args:
        config: Augmentation configuration
        sample_rate: Audio sample rate
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        sample_rate: int = 44100,
    ):
        self.config = config or AugmentationConfig()
        self.sample_rate = sample_rate
        self.pipeline = AugmentationPipeline(config, sample_rate)
    
    def __call__(self, stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply consistent augmentations to all stems.
        
        Args:
            stems: Dictionary of stem name -> audio array
            
        Returns:
            Dictionary of augmented stems
        """
        # Generate random parameters once
        random_params = self._generate_random_params()
        
        augmented_stems = {}
        for name, audio in stems.items():
            augmented_stems[name] = self._apply_with_params(audio, random_params)
        
        return augmented_stems
    
    def _generate_random_params(self) -> Dict:
        """Generate consistent random parameters."""
        return {
            'tempo_rate': random.uniform(*self.config.tempo_range) if self.config.tempo_enabled else None,
            'pitch_shift': random.uniform(*self.config.pitch_range) if self.config.pitch_enabled else None,
            'dynamic_threshold': random.uniform(-30, -10),
            'reverb_room_size': random.uniform(*self.config.reverb_room_size_range),
            'noise_type': random.choice(self.config.noise_types),
            'noise_level': random.uniform(*self.config.noise_level_range),
            'eq_gains': [random.uniform(*self.config.eq_boost_range) for _ in range(4)],
            'stereo_width': random.uniform(*self.config.stereo_width_range),
            'volume': random.uniform(*self.config.volume_range),
        }
    
    def _apply_with_params(
        self,
        audio: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Apply augmentations with pre-generated parameters."""
        # Apply each augmentation with consistent params
        # (Implementation similar to AugmentationPipeline)
        return audio  # Placeholder - would implement full augmentation logic


class VariationGenerator:
    """
    High-level interface for generating audio variations.
    Designed to easily generate 1000+ variations for training.
    
    Args:
        config: Augmentation configuration
        sample_rate: Audio sample rate
        output_dir: Directory to save variations (optional)
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        sample_rate: int = 44100,
        output_dir: Optional[str] = None,
    ):
        self.config = config or AugmentationConfig()
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.pipeline = AugmentationPipeline(config, sample_rate)
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_dataset_variations(
        self,
        audio_samples: List[Tuple[np.ndarray, Dict]],
        variations_per_sample: int = 10,
        save: bool = False,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate variations for entire dataset.
        
        Args:
            audio_samples: List of (audio, metadata) tuples
            variations_per_sample: Number of variations per original sample
            save: Whether to save variations to disk
            
        Returns:
            List of (variation_audio, variation_metadata) tuples
        """
        all_variations = []
        
        for orig_idx, (audio, metadata) in enumerate(audio_samples):
            # Generate variations
            variations = self.pipeline.generate_variations(
                audio,
                variations_per_sample,
                return_metadata=True
            )
            
            for var_idx, (var_audio, var_meta) in enumerate(variations):
                # Create new metadata
                new_metadata = metadata.copy()
                new_metadata.update({
                    'original_id': metadata.get('id', f'sample_{orig_idx}'),
                    'variation_id': var_idx,
                    'parent_sample': orig_idx,
                    'augmentation_params': var_meta,
                })
                
                all_variations.append((var_audio, new_metadata))
                
                # Save if requested
                if save and self.output_dir:
                    self._save_variation(var_audio, new_metadata, orig_idx, var_idx)
        
        logger.info(f"Generated {len(all_variations)} variations from {len(audio_samples)} samples")
        return all_variations
    
    def _save_variation(
        self,
        audio: np.ndarray,
        metadata: Dict,
        orig_idx: int,
        var_idx: int
    ):
        """Save a variation to disk."""
        import soundfile as sf
        
        orig_id = metadata.get('original_id', f'sample_{orig_idx}')
        filename = f"{orig_id}_var_{var_idx}.wav"
        filepath = Path(self.output_dir) / filename
        
        sf.write(str(filepath), audio, self.sample_rate)
        
        # Save metadata
        meta_path = filepath.with_suffix('.json')
        import json
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_all_variations(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Generate all possible variations of an audio sample.
        Useful for creating comprehensive augmentation sets.
        
        Args:
            audio: Input audio
            
        Returns:
            List of all variations
        """
        variations = []
        
        # Generate variations with different random seeds
        for i in range(self.config.variation_count):
            random.seed(self.pipeline.random_seed + i)
            np.random.seed(self.pipeline.random_seed + i)
            
            var = self.pipeline(audio)
            variations.append(var)
        
        # Reset seed
        self.pipeline.reset_seed()
        
        return variations


def create_augmentation_config(
    variation_count: int = 1000,
    enable_tempo: bool = True,
    enable_pitch: bool = True,
    enable_all: bool = True,
) -> AugmentationConfig:
    """
    Create an augmentation configuration for generating 1000+ variations.
    
    Args:
        variation_count: Number of variations to generate
        enable_tempo: Enable tempo variation
        enable_pitch: Enable pitch variation
        enable_all: Enable all augmentations
        
    Returns:
        AugmentationConfig instance
    """
    config = AugmentationConfig()
    config.variation_count = variation_count
    
    if not enable_all:
        config.tempo_enabled = enable_tempo
        config.pitch_enabled = enable_pitch
        # Keep some basic augmentations
        config.noise_enabled = True
        config.volume_enabled = True
    else:
        # All augmentations enabled with wide ranges
        config.tempo_range = (0.7, 1.3)  # ±30%
        config.pitch_range = (-5.0, 5.0)  # ±5 semitones
        config.noise_level_range = (0.0005, 0.03)
        config.eq_boost_range = (-10.0, 10.0)
        config.reverb_room_size_range = (0.1, 1.0)
        config.stereo_width_range = (0.3, 1.7)
        config.volume_range = (0.5, 1.0)
    
    return config

