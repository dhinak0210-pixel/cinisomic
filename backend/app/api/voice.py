"""
Voice cloning API endpoints.
"""
import os
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.security import get_current_user
from app.models.database import VoiceSample, User
from app.models.schemas import (
    VoiceSampleUpload,
    VoiceSampleResponse,
    VoiceCloneRequest,
    ErrorResponse
)
from app.core.database import get_db

router = APIRouter(prefix="/voice", tags=["Voice Cloning"])


@router.post(
    "/upload-sample",
    response_model=VoiceSampleResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_voice_sample(
    file: UploadFile = File(...),
    name: str = Form(...),
    consent: bool = Form(True),
    language: str = Form("en"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a voice sample for cloning.
    
    - **file**: Audio file (WAV, MP3, FLAC)
    - **name**: Name for the voice sample
    - **consent**: User consent for using voice data
    - **language**: Language of the voice sample
    """
    # Validate consent
    if not consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Consent is required to upload voice samples"
        )
    
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create upload directory
    user_voice_dir = os.path.join(settings.voice_samples_dir, str(current_user.id))
    os.makedirs(user_voice_dir, exist_ok=True)
    
    # Generate unique filename
    sample_id = uuid.uuid4()
    filename = f"{sample_id}{file_ext}"
    filepath = os.path.join(user_voice_dir, filename)
    
    # Save file
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    # Create database record
    voice_sample = VoiceSample(
        user_id=current_user.id,
        name=name,
        file_path=filepath,
        consent_given=consent,
        consent_timestamp=datetime.utcnow(),
        language=language,
        is_processed=False
    )
    
    db.add(voice_sample)
    await db.commit()
    await db.refresh(voice_sample)
    
    return voice_sample


@router.get("/samples", response_model=list[VoiceSampleResponse])
async def list_voice_samples(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all voice samples for the current user.
    """
    result = await db.execute(
        select(VoiceSample)
        .where(VoiceSample.user_id == current_user.id)
        .order_by(VoiceSample.created_at.desc())
    )
    samples = result.scalars().all()
    return samples


@router.get("/samples/{sample_id}", response_model=VoiceSampleResponse)
async def get_voice_sample(
    sample_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific voice sample.
    """
    result = await db.execute(
        select(VoiceSample).where(
            VoiceSample.id == sample_id,
            VoiceSample.user_id == current_user.id
        )
    )
    sample = result.scalar_one_or_none()
    
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice sample not found"
        )
    
    return sample


@router.delete("/samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice_sample(
    sample_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a voice sample.
    """
    result = await db.execute(
        select(VoiceSample).where(
            VoiceSample.id == sample_id,
            VoiceSample.user_id == current_user.id
        )
    )
    sample = result.scalar_one_or_none()
    
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice sample not found"
        )
    
    # Delete file
    if os.path.exists(sample.file_path):
        os.remove(sample.file_path)
    
    # Delete embedding if exists
    if sample.embedding_path and os.path.exists(sample.embedding_path):
        os.remove(sample.embedding_path)
    
    await db.delete(sample)
    await db.commit()


@router.post("/generate")
async def generate_with_cloned_voice(
    request: VoiceCloneRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate audio using a cloned voice.
    This would integrate with the music generation service.
    """
    # Verify voice sample exists and belongs to user
    result = await db.execute(
        select(VoiceSample).where(
            VoiceSample.id == request.voice_sample_id,
            VoiceSample.user_id == current_user.id,
            VoiceSample.is_processed == True
        )
    )
    sample = result.scalar_one_or_none()
    
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice sample not found or not processed yet"
        )
    
    # For now, return a placeholder response
    # In production, this would call the voice cloning service
    return {
        "message": "Voice cloning generation started",
        "voice_sample_id": request.voice_sample_id,
        "lyrics": request.lyrics[:100] + "...",
        "status": "processing"
    }

