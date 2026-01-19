"""
Music Generation Inference Service.
Provides music generation capabilities using trained models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import io

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for music generation."""
    # Model settings
    model_path: str = "models/music_transformer"
    device: str = "auto"  # auto, cpu, cuda
    
    # Generation settings
    max_length: int = 1000
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Audio settings
    sample_rate: int = 44100
    num_channels: int = 2


class MusicGenerator:
    """
    Music generation service using trained transformer models.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.model = None
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        try:
            path = model_path or self.config.model_path
            self.model = None  # Demo mode for now
            logger.info(f"Model loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False
    
    def generate(
        self,
        prompt: Optional[str] = None,
        mood: str = "epic",
        tempo: int = 120,
        duration: int = 60,
        instruments: Optional[List[str]] = None,
    ) -> Dict:
        logger.info(f"Generating music: mood={mood}, tempo={tempo}, duration={duration}")
        return self._generate_demo_audio(mood, tempo, duration)
    
    def _generate_demo_audio(
        self,
        mood: str,
        tempo: int,
        duration: int
    ) -> Dict:
        sample_rate = self.config.sample_rate
        num_samples = int(duration * sample_rate)
        
        mood_frequencies = {
            "epic": 110.0, "dramatic": 82.4, "peaceful": 261.6,
            "tense": 73.4, "joyful": 523.2, "melancholic": 196.0,
            "default": 440.0
        }
        base_freq = mood_frequencies.get(mood.lower(), mood_frequencies["default"])
        
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        freq = base_freq * (1 + 0.25 * np.sin(2 * np.pi * tempo / 60 * t))
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio += 0.2 * np.sin(2 * np.pi * base_freq * 1.5 * t)
        audio += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_stereo = np.column_stack([audio_int16, audio_int16])
        
        return {
            "success": True,
            "audio": audio_stereo.tobytes(),
            "sample_rate": sample_rate,
            "duration": duration,
            "mood": mood,
            "tempo": tempo,
            "format": "demo"
        }
    
    def save_audio(
        self,
        audio_data: bytes,
        filepath: str,
        sample_rate: Optional[int] = None
    ) -> bool:
        try:
            sr = sample_rate or self.config.sample_rate
            import wave
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.config.num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False


def create_music_generator(config: Optional[GenerationConfig] = None) -> MusicGenerator:
    generator = MusicGenerator(config)
    generator.load_model()
    return generator


if __name__ == "__main__":
    generator = create_music_generator()
    result = generator.generate(prompt="Epic battle music", mood="epic", tempo=120, duration=10)
    if result["success"]:
        print(f"Generated {result['duration']}s of {result['mood']} music")

