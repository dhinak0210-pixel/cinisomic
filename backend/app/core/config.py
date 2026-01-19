"""
Backend configuration settings for Cinematic Music Platform.
"""
import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # App
    app_name: str = "Cinematic Music Platform API"
    app_version: str = "1.0.0"
    debug: bool = True
    api_prefix: str = "/api/v1"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./cinematic_music.db"
    
    @property
    def async_database_url(self) -> str:
        """Get async database URL, converting from sync if needed."""
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("sqlite://"):
            url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return url
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Celery (task queue)
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # File storage
    uploads_dir: str = "./uploads"
    generated_audio_dir: str = "./generated_audio"
    voice_samples_dir: str = "./voice_samples"
    
    # Audio settings
    default_sample_rate: int = 44100
    default_channels: int = 2
    max_duration_seconds: int = 300
    max_file_size_mb: int = 100
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]
    
    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from environment variable if it's a string."""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return self.cors_origins
    
    # AI Services
    model_path: str = "../ai-services/training/models"
    device: str = "auto"
    
    # External API Keys
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""
    suno_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()

