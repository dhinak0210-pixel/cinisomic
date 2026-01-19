"""
Database models for Cinematic Music Platform.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class GenerationType(str, enum.Enum):
    """Type of music generation."""
    BGM = "bgm"
    SONG = "song"


class TaskStatus(str, enum.Enum):
    """Status of a generation task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    """User model for authentication and preferences."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Preferences
    preferences = Column(JSON, default=dict)
    
    # Relationships
    projects = relationship("Project", back_populates="owner")
    voice_samples = relationship("VoiceSample", back_populates="user")
    generation_tasks = relationship("GenerationTask", back_populates="user")


class Project(Base):
    """Project model for organizing generated music."""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Project settings
    settings = Column(JSON, default=dict)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    tracks = relationship("Track", back_populates="project")
    scenes = relationship("Scene", back_populates="project")


class Track(Base):
    """Track model for individual audio tracks."""
    __tablename__ = "tracks"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    track_type = Column(String(50))
    
    # Audio file info
    file_path = Column(String(512))
    duration = Column(Float, default=0.0)
    sample_rate = Column(Integer, default=44100)
    
    # Generation params
    generation_params = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="tracks")


class Scene(Base):
    """Scene model for film timeline."""
    __tablename__ = "scenes"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    start_time = Column(Float, default=0.0)
    duration = Column(Float, default=0.0)
    
    # Scene metadata
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="scenes")


class VoiceSample(Base):
    """Voice sample model for voice cloning."""
    __tablename__ = "voice_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    duration = Column(Float, default=0.0)
    
    # Consent and processing
    consent_given = Column(Boolean, default=True)
    consent_timestamp = Column(DateTime)
    is_processed = Column(Boolean, default=False)
    embedding_path = Column(String(512))
    
    # Sample metadata
    sample_rate = Column(Integer, default=44100)
    language = Column(String(10), default="en")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="voice_samples")


class GenerationTask(Base):
    """Model for tracking generation tasks."""
    __tablename__ = "generation_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(36), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Task details
    generation_type = Column(String(50), nullable=False)
    status = Column(String(20), default=TaskStatus.PENDING)
    progress = Column(Integer, default=0)
    
    # Input parameters
    input_params = Column(JSON, default=dict)
    
    # Output
    output_path = Column(String(512))
    output_metadata = Column(JSON, default=dict)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="generation_tasks")


class LyricsProject(Base):
    """Model for lyrics projects."""
    __tablename__ = "lyrics_projects"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    lyrics = Column(Text)
    language = Column(String(10), default="en")
    
    # Sync data (timestamps for each line)
    sync_data = Column(JSON, default=list)
    
    # Generation params
    generation_params = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

