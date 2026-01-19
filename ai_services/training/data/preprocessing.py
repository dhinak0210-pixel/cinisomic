"""
Audio preprocessing utilities for cinematic music generation.
Provides functions for audio normalization, feature extraction, and audio transformations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for cinematic music.
    Handles resampling, normalization, silence removal, and format conversion.
    
    Args:
        sample_rate: Target sample rate
        normalize: Whether to normalize audio
        remove_silence: Whether to remove silence
        silence_threshold: Threshold for silence detection in dB
        target_duration: Target duration in seconds (None for no cropping)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        normalize: bool = True,
        remove_silence: bool = True,
        silence_threshold: float = -40,
        target_duration: Optional[float] = None,
        target_channels: int = 2,
    ):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.target_duration = target_duration
        self.target_channels = target_channels
        
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            logger.warning("librosa not available, using basic audio processing")
            self.librosa_available = False
    
    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply preprocessing to audio.
        
        Args:
            audio: Input audio waveform
            sr: Original sample rate
            
        Returns:
            Preprocessed audio waveform
        """
        # Resample if necessary
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)
        
        # Convert to target channels
        if len(audio.shape) == 1:
            audio = np.tile(audio, (self.target_channels, 1))
        elif audio.shape[0] != self.target_channels:
            if audio.shape[0] < self.target_channels:
                audio = np.pad(audio, ((0, self.target_channels - audio.shape[0]), (0, 0)))
            else:
                audio = audio[:self.target_channels, :]
        
        # Remove silence
        if self.remove_silence:
            audio = self._remove_silence(audio)
        
        # Normalize
        if self.normalize:
            audio = self._normalize(audio)
        
        # Crop to target duration
        if self.target_duration is not None:
            audio = self._crop_to_duration(audio, self.target_duration)
        
        return audio
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if self.librosa_available:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            # Basic linear interpolation resampling
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silence from beginning and end of audio."""
        if self.librosa_available:
            import librosa
            # For multi-channel, process each channel
            if len(audio.shape) > 1:
                processed = []
                for channel in audio:
                    intervals = librosa.effects.split(
                        channel, top_db=-self.silence_threshold
                    )
                    if len(intervals) > 0:
                        channel = np.concatenate([channel[start:end] for start, end in intervals])
                    processed.append(channel)
                
                # Pad to same length
                max_len = max(c.shape[0] for c in processed)
                for i in range(len(processed)):
                    if processed[i].shape[0] < max_len:
                        processed[i] = np.pad(processed[i], (0, max_len - processed[i].shape[0]))
                
                return np.array(processed)
            else:
                intervals = librosa.effects.split(
                    audio, top_db=-self.silence_threshold
                )
                if len(intervals) > 0:
                    return np.concatenate([audio[start:end] for start, end in intervals])
                return audio
        return audio
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _crop_to_duration(self, audio: np.ndarray, duration: float) -> np.ndarray:
        """Crop audio to specified duration."""
        target_length = int(duration * self.sample_rate)
        if len(audio.shape) == 1:
            if len(audio) > target_length:
                start = np.random.randint(0, len(audio) - target_length)
                return audio[start:start + target_length]
            else:
                return np.pad(audio, (0, target_length - len(audio)))
        else:
            if audio.shape[1] > target_length:
                start = np.random.randint(0, audio.shape[1] - target_length)
                return audio[:, start:start + target_length]
            else:
                return np.pad(audio, ((0, 0), (0, target_length - audio.shape[1])))


class FeatureExtractor:
    """
    Feature extraction for audio analysis.
    Extracts various audio features for training and evaluation.
    
    Args:
        sample_rate: Sample rate for feature extraction
        n_mels: Number of mel bins for spectrogram
        n_fft: FFT window size
        hop_length: Hop length between frames
        win_length: Window length
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            self.librosa_available = False
            logger.warning("librosa not available, feature extraction limited")
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Mel spectrogram
        """
        if not self.librosa_available:
            raise RuntimeError("librosa required for mel spectrogram extraction")
        
        import librosa
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0,
            fmax=self.sample_rate / 2,
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract short-time Fourier transform spectrogram.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Spectrogram
        """
        if not self.librosa_available:
            raise RuntimeError("librosa required for spectrogram extraction")
        
        import librosa
        
        stft = librosa.stft(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        
        # Get magnitude
        spectrogram = np.abs(stft)
        
        # Convert to log scale
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        return spectrogram_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Input audio waveform
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features
        """
        if not self.librosa_available:
            raise RuntimeError("librosa required for MFCC extraction")
        
        import librosa
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        return mfcc
    
    def extract_features_dict(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features and return as dictionary.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Mel spectrogram
        try:
            features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        except Exception as e:
            logger.warning(f"Failed to extract mel spectrogram: {e}")
        
        # Spectrogram
        try:
            features['spectrogram'] = self.extract_spectrogram(audio)
        except Exception as e:
            logger.warning(f"Failed to extract spectrogram: {e}")
        
        # MFCC
        try:
            features['mfcc'] = self.extract_mfcc(audio)
        except Exception as e:
            logger.warning(f"Failed to extract MFCC: {e}")
        
        # Additional features
        try:
            import librosa
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate
            )
            features['rms'] = librosa.feature.rms(y=audio)
        except Exception as e:
            logger.warning(f"Failed to extract additional features: {e}")
        
        return features
    
    def features_to_tensor(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert numpy features to torch tensors.
        
        Args:
            features: Dictionary of numpy arrays
            
        Returns:
            Dictionary of torch tensors
        """
        tensor_features = {}
        for name, arr in features.items():
            tensor_features[name] = torch.from_numpy(arr).float()
        return tensor_features
    
    def audio_to_mel_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """
        Convert audio directly to mel spectrogram tensor.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Mel spectrogram tensor
        """
        mel_spec = self.extract_mel_spectrogram(audio)
        return torch.from_numpy(mel_spec).float()


class AudioTransform:
    """Base class for audio transformations."""
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TimeStretch(AudioTransform):
    """
    Time stretching transformation.
    Changes the speed of audio without changing pitch.
    
    Args:
        rate: Stretching rate (0.5 = half speed, 2.0 = double speed)
    """
    
    def __init__(self, rate: float = 1.0):
        self.rate = rate
        
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            self.librosa_available = False
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if not self.librosa_available:
            logger.warning("librosa required for time stretching")
            return audio
        
        import librosa
        
        if len(audio.shape) > 1:
            stretched = []
            for channel in audio:
                stretched.append(librosa.effects.time_stretch(channel, rate=self.rate))
            return np.array(stretched)
        else:
            return librosa.effects.time_stretch(audio, rate=self.rate)


class PitchShift(AudioTransform):
    """
    Pitch shifting transformation.
    Changes the pitch of audio without changing speed.
    
    Args:
        n_steps: Number of semitones to shift (positive = higher, negative = lower)
        sr: Sample rate
    """
    
    def __init__(self, n_steps: float, sr: int = 44100):
        self.n_steps = n_steps
        self.sr = sr
        
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            self.librosa_available = False
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if not self.librosa_available:
            logger.warning("librosa required for pitch shifting")
            return audio
        
        import librosa
        
        if len(audio.shape) > 1:
            shifted = []
            for channel in audio:
                shifted.append(librosa.effects.pitch_shift(channel, sr=self.sr, n_steps=self.n_steps))
            return np.array(shifted)
        else:
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=self.n_steps)


class DynamicRangeCompression(AudioTransform):
    """
    Dynamic range compression.
    Reduces the difference between loud and quiet parts.
    
    Args:
        threshold: Threshold in dB
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
    """
    
    def __init__(
        self,
        threshold: float = -20,
        ratio: float = 4.0,
        attack: float = 0.003,
        release: float = 0.25,
    ):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        # Simple implementation using peak normalization
        threshold = 10 ** (self.threshold / 20)
        
        # Find segments above threshold
        abs_audio = np.abs(audio)
        above_threshold = abs_audio > threshold
        
        if not np.any(above_threshold):
            return audio
        
        # Apply compression
        compressed = audio.copy()
        mask = above_threshold
        
        # Calculate gain reduction
        overshoot = (abs_audio[mask] - threshold) / abs_audio[mask]
        gain_reduction = 1 - (overshoot * (1 - 1/self.ratio))
        
        compressed[mask] = audio[mask] * np.clip(gain_reduction, 0, 1)
        
        return compressed


class ReverbEffect(AudioTransform):
    """
    Simple reverb effect using convolution.
    
    Args:
        room_size: Room size (affects reverb duration)
        damping: High-frequency damping
        wet_level: Level of reverb signal
        dry_level: Level of original signal
    """
    
    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7,
        sample_rate: int = 44100,
    ):
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
        self.dry_level = dry_level
        self.sample_rate = sample_rate
        
        # Create impulse response
        self.ir = self._create_impulse_response()
    
    def _create_impulse_response(self) -> np.ndarray:
        """Create synthetic impulse response for reverb."""
        length = int(self.sample_rate * self.room_size * 2)
        ir = np.zeros(length)
        
        # Exponential decay
        decay = np.exp(-np.linspace(0, 10, length))
        
        # Add some random noise for diffusion
        noise = np.random.randn(length) * 0.1
        
        # Combine
        ir = decay * (1 - self.damping + noise * self.damping)
        
        # Normalize
        ir = ir / np.max(np.abs(ir)) if np.max(np.abs(ir)) > 0 else ir
        
        return ir
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import convolve
            import numpy as np
            
            if len(audio.shape) > 1:
                reverb = []
                for channel in audio:
                    reverb.append(convolve(channel, self.ir, mode='same'))
                reverb = np.array(reverb)
            else:
                reverb = convolve(audio, self.ir, mode='same')
            
            # Mix dry and wet
            return self.dry_level * audio + self.wet_level * reverb
            
        except ImportError:
            logger.warning("scipy required for reverb effect")
            return audio


class NoiseInjection(AudioTransform):
    """
    Add random noise to audio.
    Useful for data augmentation.
    
    Args:
        noise_level: Maximum noise level as fraction of signal (0-1)
        noise_type: Type of noise ('white', 'pink', 'brown')
    """
    
    def __init__(
        self,
        noise_level: float = 0.005,
        noise_type: str = 'white',
    ):
        self.noise_level = noise_level
        self.noise_type = noise_type
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        noise_level = np.random.uniform(0, self.noise_level)
        
        if self.noise_type == 'white':
            noise = np.random.randn(*audio.shape).astype(np.float32)
        elif self.noise_type == 'pink':
            noise = self._pink_noise(audio.shape)
        elif self.noise_type == 'brown':
            noise = self._brown_noise(audio.shape)
        else:
            noise = np.random.randn(*audio.shape).astype(np.float32)
        
        # Scale noise by signal level
        signal_level = np.max(np.abs(audio))
        if signal_level > 0:
            noise = noise * signal_level * noise_level
        else:
            noise = noise * noise_level
        
        return audio + noise
    
    def _pink_noise(self, shape: tuple) -> np.ndarray:
        """Generate pink noise."""
        white = np.random.randn(*shape).astype(np.float32)
        # Approximate pink noise using 1/f filter
        pink = np.cumsum(white) / np.sqrt(np.arange(1, shape[0] + 1))
        return pink / np.max(np.abs(pink))
    
    def _brown_noise(self, shape: tuple) -> np.ndarray:
        """Generate brown (red) noise."""
        white = np.random.randn(*shape).astype(np.float32)
        # Integrate white noise
        brown = np.cumsum(white)
        brown = brown / np.max(np.abs(brown)) if np.max(np.abs(brown)) > 0 else brown
        return brown


def mix_augmentations(
    audio: np.ndarray,
    augmentations: list,
    p: float = 0.5
) -> np.ndarray:
    """
    Apply random augmentations with probability p.
    
    Args:
        audio: Input audio
        augmentations: List of augmentation transforms
        p: Probability of applying each augmentation
        
    Returns:
        Augmented audio
    """
    augmented = audio.copy()
    
    for aug in augmentations:
        if np.random.random() < p:
            augmented = aug(augmented)
    
    return augmented


def create_preprocessing_pipeline(
    sample_rate: int = 44100,
    normalize: bool = True,
    remove_silence: bool = True,
    augmentations: Optional[list] = None,
) -> Tuple[AudioPreprocessor, list]:
    """
    Create a preprocessing pipeline.
    
    Args:
        sample_rate: Target sample rate
        normalize: Whether to normalize
        remove_silence: Whether to remove silence
        augmentations: Optional list of augmentations
        
    Returns:
        Tuple of (preprocessor, augmentations list)
    """
    preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        normalize=normalize,
        remove_silence=remove_silence,
    )
    
    if augmentations is None:
        augmentations = [
            TimeStretch(rate=np.random.uniform(0.9, 1.1)),
            PitchShift(n_steps=np.random.uniform(-2, 2), sr=sample_rate),
            DynamicRangeCompression(),
            NoiseInjection(noise_level=0.01),
        ]
    
    return preprocessor, augmentations

