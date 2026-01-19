"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from enum import Enum


# ============ Auth Schemas ============

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for access token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Schema for token payload."""
    user_id: Optional[int] = None
    email: Optional[str] = None


# ============ Music Generation Schemas ============

class MusicGenerationRequest(BaseModel):
    """Schema for music generation request."""
    prompt: str = Field(..., min_length=10, max_length=1000)
    mood: str = "epic"
    tempo: int = Field(default=120, ge=60, le=180)
    duration: int = Field(default=60, ge=15, le=300)
    instruments: Optional[List[str]] = None
    generation_type: str = "bgm"
    project_id: Optional[int] = None


class SongGenerationRequest(BaseModel):
    """Schema for song generation from lyrics."""
    lyrics: str = Field(..., min_length=50, max_length=5000)
    voice: str = "default"
    mood: str = "epic"
    tempo: int = Field(default=120, ge=60, le=180)
    duration: Optional[int] = None
    language: str = "en"
    project_id: Optional[int] = None


class GenerationProgress(BaseModel):
    """Schema for generation progress response."""
    task_id: str
    status: str
    progress: int
    message: str


class GenerationResponse(BaseModel):
    """Schema for generation result response."""
    task_id: str
    success: bool
    audio_url: Optional[str]
    duration: float
    sample_rate: int
    metadata: Dict[str, Any]


# ============ Voice Cloning Schemas ============

class VoiceSampleUpload(BaseModel):
    """Schema for voice sample upload."""
    name: str = Field(..., min_length=1, max_length=100)
    consent: bool = True
    language: str = "en"


class VoiceSampleResponse(BaseModel):
    """Schema for voice sample response."""
    id: int
    name: str
    duration: float
    is_processed: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class VoiceCloneRequest(BaseModel):
    """Schema for voice cloning generation request."""
    lyrics: str
    voice_sample_id: int
    mood: str = "neutral"
    tempo: int = Field(default=120, ge=60, le=180)
    emotion: str = "neutral"


# ============ Lyrics Schemas ============

class LyricsGenerationRequest(BaseModel):
    """Schema for lyrics generation request."""
    prompt: str = Field(..., min_length=20, max_length=500)
    style: str = "cinematic"
    mood: str = "epic"
    language: str = "en"
    length: int = Field(default=200, ge=50, le=500)


class LyricsSyncRequest(BaseModel):
    """Schema for lyrics synchronization request."""
    lyrics_id: int
    sync_data: List[Dict[str, Any]]  # [{"line": "...", "start_time": 0, "end_time": 5}]


class LyricsProjectResponse(BaseModel):
    """Schema for lyrics project response."""
    id: int
    title: str
    lyrics: Optional[str]
    language: str
    sync_data: List[Dict]
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Project Schemas ============

class ProjectCreate(BaseModel):
    """Schema for creating a project."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ProjectResponse(BaseModel):
    """Schema for project response."""
    id: int
    name: str
    description: Optional[str]
    owner_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============ Scene Schemas ============

class SceneCreate(BaseModel):
    """Schema for creating a scene."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    start_time: float = 0.0
    duration: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class SceneResponse(BaseModel):
    """Schema for scene response."""
    id: int
    project_id: int
    name: str
    description: Optional[str]
    start_time: float
    duration: float
    metadata: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Export Schemas ============

class ExportRequest(BaseModel):
    """Schema for export request."""
    task_id: str
    format: str = Field(default="mp3", pattern="^(mp3|wav|midi|stems)$")
    quality: Optional[str] = None  # "high", "medium", "low"


class ExportResponse(BaseModel):
    """Schema for export response."""
    download_url: str
    expires_at: datetime
    file_size: int


# ============ Error Schemas ============

class ErrorResponse(BaseModel):
    """Schema for error response."""
    detail: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ValidationErrorResponse(BaseModel):
    """Schema for validation error response."""
    detail: List[Dict[str, Any]]

