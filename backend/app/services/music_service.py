"""
Music generation service with task queue support.
"""
import asyncio
import uuid
import os
import sys
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import numpy as np

from app.core.config import settings
from app.models.database import GenerationTask, TaskStatus

# Add project root to path for ai_services imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_services.inference.music_generator import MusicGenerator, GenerationConfig


# Global music generator instance
_music_generator: Optional[MusicGenerator] = None


def get_music_generator() -> MusicGenerator:
    """Get or create the music generator instance."""
    global _music_generator
    if _music_generator is None:
        config = GenerationConfig(
            model_path=settings.model_path,
            device=settings.device,
            sample_rate=settings.default_sample_rate
        )
        _music_generator = MusicGenerator(config)
        _music_generator.load_model()
    return _music_generator


async def generate_music_task(
    task_id: str,
    user_id: int,
    prompt: Optional[str],
    mood: str,
    tempo: int,
    duration: int,
    instruments: List[str],
    lyrics: Optional[str] = None,
    voice: Optional[str] = None,
    language: str = "en"
):
    """
    Background task for music generation.
    
    This function is designed to run in a background task queue.
    """
    from app.core.database import get_db
    from app.models.database import GenerationTask, TaskStatus
    
    # Note: In a production system, this would use Celery or similar.
    # For now, we create a new database session for this task.
    
    try:
        # Import here to avoid circular imports
        from app.core.database import AsyncSessionLocal
        
        async with AsyncSessionLocal() as db:
            # Update task status to processing
            result = await db.execute(
                select(GenerationTask).where(GenerationTask.task_id == task_id)
            )
            task = result.scalar_one_or_none()
            
            if not task:
                return
            
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()
            await db.commit()
            
            # Update progress
            task.progress = 10
            await db.commit()
            
            # Generate music
            generator = get_music_generator()
            
            # Update progress
            task.progress = 30
            await db.commit()
            
            # Perform generation
            if lyrics:
                # Song generation
                result = generator.generate(
                    prompt=f"Song with lyrics: {lyrics[:100]}...",
                    mood=mood,
                    tempo=tempo,
                    duration=duration,
                    instruments=instruments
                )
            else:
                # BGM generation
                result = generator.generate(
                    prompt=prompt,
                    mood=mood,
                    tempo=tempo,
                    duration=duration,
                    instruments=instruments
                )
            
            # Update progress
            task.progress = 70
            await db.commit()
            
            # Save the generated audio
            if result.get("success"):
                # Create output directory if needed
                output_dir = settings.generated_audio_dir
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename
                filename = f"{user_id}_{task_id}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # Save audio
                if "audio" in result:
                    generator.save_audio(
                        result["audio"],
                        filepath,
                        result.get("sample_rate", settings.default_sample_rate)
                    )
                    
                    # Update task with output
                    task.output_path = f"/api/v1/export/download/{filename}"
                    task.output_metadata = {
                        "duration": result.get("duration", duration),
                        "sample_rate": result.get("sample_rate", settings.default_sample_rate),
                        "mood": mood,
                        "tempo": tempo,
                        "format": result.get("format", "wav")
                    }
                    
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100
                    task.completed_at = datetime.utcnow()
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = "No audio data generated"
            else:
                task.status = TaskStatus.FAILED
                task.error_message = result.get("error", "Unknown error")
            
            await db.commit()
            
    except Exception as e:
        # Log the error and update task status
        try:
            from app.core.database import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(GenerationTask).where(GenerationTask.task_id == task_id)
                )
                task = result.scalar_one_or_none()
                if task:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    await db.commit()
        except Exception:
            pass  # Best effort cleanup
        
        print(f"Music generation task failed: {task_id}, error: {e}")


class SunoApiClient:
    """Suno API client for music generation."""
    
    BASE_URL = "https://api.suno.ai/v1"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.suno_api_key
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def generate_music(
        self,
        prompt: str,
        lyrics: str = None,
        style: str = "cinematic",
        duration: int = 60,
        instrumental: bool = False
    ) -> dict:
        """
        Generate music using Suno API.
        
        Args:
            prompt: Description of the music to generate
            lyrics: Song lyrics (for vocal tracks)
            style: Music style (cinematic, epic, dramatic, etc.)
            duration: Duration in seconds
            instrumental: Whether to generate instrumental track
            
        Returns:
            dict with generation result including task_id for polling
        """
        if not self.api_key:
            return {"success": False, "error": "Suno API key not configured"}
        
        try:
            # Suno API endpoint for generation
            url = f"{self.BASE_URL}/generate"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "style": style,
                "duration": min(duration, 180),  # Max 180 seconds
                "instrumental": instrumental,
                "title": ""
            }
            
            if lyrics and not instrumental:
                payload["lyrics"] = lyrics
            
            response = await self.client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "task_id": data.get("id"),
                    "status": "pending"
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_generation_status(self, task_id: str) -> dict:
        """Check the status of a generation task."""
        if not self.api_key:
            return {"success": False, "error": "Suno API key not configured"}
        
        try:
            url = f"{self.BASE_URL}/generate/{task_id}"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = await self.client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "status": data.get("status"),
                    "audio_url": data.get("audio_url"),
                    "video_url": data.get("video_url"),
                    "title": data.get("title")
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global Suno client instance
_suno_client: Optional[SunoApiClient] = None


def get_suno_client() -> SunoApiClient:
    """Get or create the Suno API client instance."""
    global _suno_client
    if _suno_client is None:
        _suno_client = SunoApiClient()
    return _suno_client


# Demo mode fallback function
def generate_demo_music(
    mood: str,
    tempo: int,
    duration: int
) -> dict:
    """Generate demo audio using simple synthesis."""
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    # Map moods to base frequencies
    mood_frequencies = {
        "epic": 110.0,
        "dramatic": 82.4,
        "peaceful": 261.6,
        "tense": 73.4,
        "joyful": 523.2,
        "melancholic": 196.0,
        "romantic": 293.7,
        "mysterious": 146.8,
        "heroic": 130.8,
        "default": 440.0
    }
    
    base_freq = mood_frequencies.get(mood.lower(), mood_frequencies["default"])
    
    # Generate audio waveform
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    
    # Main melodic component
    freq = base_freq * (1 + 0.25 * np.sin(2 * np.pi * tempo / 60 * t))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # Harmonics
    audio += 0.2 * np.sin(2 * np.pi * base_freq * 1.5 * t)
    audio += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
    
    # Add some variation based on tempo
    if tempo > 100:
        # More energetic - add some rhythmic elements
        rhythm_freq = tempo / 60
        rhythm = 0.1 * np.sin(2 * np.pi * rhythm_freq * t)
        audio += rhythm
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to 16-bit integer
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create stereo
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

