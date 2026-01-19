"""
Music generation API endpoints.
"""
import uuid
import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.security import get_current_user, rate_limiter
from app.core.database import get_db
from app.models.database import GenerationTask, TaskStatus, User, VoiceSample
from app.models.schemas import (
    MusicGenerationRequest,
    SongGenerationRequest,
    GenerationProgress,
    GenerationResponse,
    ErrorResponse
)
from app.services.music_service import (
    generate_music_task,
    get_suno_client,
    generate_demo_music
)

router = APIRouter(prefix="/music", tags=["Music Generation"])


@router.post(
    "/generate",
    response_model=GenerationProgress,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"}
    }
)
async def generate_music(
    request: MusicGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate cinematic background music based on a text prompt.
    
    - **prompt**: Detailed description of the music to generate
    - **mood**: Emotional tone (epic, dramatic, peaceful, tense, etc.)
    - **tempo**: Beats per minute (60-180)
    - **duration**: Length in seconds (15-300)
    - **instruments**: Optional list of instruments
    """
    # Rate limiting check
    client_id = str(current_user.id)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Validate duration against max
    if request.duration > settings.max_duration_seconds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Duration cannot exceed {settings.max_duration_seconds} seconds"
        )
    
    # Try Suno API first if configured
    suno_client = get_suno_client()
    if settings.suno_api_key:
        suno_result = await suno_client.generate_music(
            prompt=request.prompt,
            style=request.mood,
            duration=request.duration,
            instrumental=True
        )
        
        if suno_result.get("success"):
            # Create task with Suno task_id
            task_id = suno_result.get("task_id", str(uuid.uuid4()))
            task = GenerationTask(
                task_id=task_id,
                user_id=current_user.id,
                generation_type="bgm",
                status=TaskStatus.PROCESSING,
                input_params={
                    "prompt": request.prompt,
                    "mood": request.mood,
                    "tempo": request.tempo,
                    "duration": request.duration,
                    "instruments": request.instruments or [],
                    "suno_task_id": suno_result.get("task_id")
                },
                progress=50
            )
            db.add(task)
            await db.commit()
            
            return GenerationProgress(
                task_id=task_id,
                status="processing",
                progress=50,
                message="Suno API generation started"
            )
    
    # Fallback to demo mode or local generation
    task_id = str(uuid.uuid4())
    task = GenerationTask(
        task_id=task_id,
        user_id=current_user.id,
        generation_type="bgm",
        status=TaskStatus.PENDING,
        input_params={
            "prompt": request.prompt,
            "mood": request.mood,
            "tempo": request.tempo,
            "duration": request.duration,
            "instruments": request.instruments or []
        }
    )
    db.add(task)
    await db.commit()
    
    # Start generation in background
    background_tasks.add_task(
        generate_music_task,
        task_id=task_id,
        user_id=current_user.id,
        prompt=request.prompt,
        mood=request.mood,
        tempo=request.tempo,
        duration=request.duration,
        instruments=request.instruments or []
    )
    
    return GenerationProgress(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Generation task queued"
    )


@router.post(
    "/generate-song",
    response_model=GenerationProgress,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"}
    }
)
async def generate_song(
    request: SongGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a song from lyrics with AI vocals.
    
    - **lyrics**: The lyrics for the song
    - **voice**: Voice selection (default or cloned voice ID)
    - **mood**: Emotional tone
    - **tempo**: Beats per minute (60-180)
    - **language**: Language code (en, es, fr, etc.)
    """
    # Rate limiting check
    client_id = str(current_user.id)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Validate voice access if using cloned voice
    if request.voice != "default":
        result = await db.execute(
            select(VoiceSample).where(
                VoiceSample.id == int(request.voice),
                VoiceSample.user_id == current_user.id
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice sample not found"
            )
    
    # Try Suno API first if configured
    suno_client = get_suno_client()
    if settings.suno_api_key:
        suno_result = await suno_client.generate_music(
            prompt=f"Song with mood: {request.mood}",
            lyrics=request.lyrics,
            style=request.mood,
            duration=request.duration or 180,
            instrumental=False
        )
        
        if suno_result.get("success"):
            task_id = suno_result.get("task_id", str(uuid.uuid4()))
            task = GenerationTask(
                task_id=task_id,
                user_id=current_user.id,
                generation_type="song",
                status=TaskStatus.PROCESSING,
                input_params={
                    "lyrics": request.lyrics,
                    "voice": request.voice,
                    "mood": request.mood,
                    "tempo": request.tempo,
                    "duration": request.duration,
                    "language": request.language,
                    "suno_task_id": suno_result.get("task_id")
                },
                progress=50
            )
            db.add(task)
            await db.commit()
            
            return GenerationProgress(
                task_id=task_id,
                status="processing",
                progress=50,
                message="Suno API song generation started"
            )
    
    # Fallback to demo mode
    task_id = str(uuid.uuid4())
    task = GenerationTask(
        task_id=task_id,
        user_id=current_user.id,
        generation_type="song",
        status=TaskStatus.PENDING,
        input_params={
            "lyrics": request.lyrics,
            "voice": request.voice,
            "mood": request.mood,
            "tempo": request.tempo,
            "duration": request.duration,
            "language": request.language
        }
    )
    db.add(task)
    await db.commit()
    
    # Start generation in background
    background_tasks.add_task(
        generate_music_task,
        task_id=task_id,
        user_id=current_user.id,
        prompt=None,
        mood=request.mood,
        tempo=request.tempo,
        duration=request.duration or 180,
        instruments=[],
        lyrics=request.lyrics,
        voice=request.voice,
        language=request.language
    )
    
    return GenerationProgress(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Song generation task queued"
    )


@router.get(
    "/result/{task_id}",
    response_model=GenerationResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Task not found"}
    }
)
async def get_generation_result(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the result of a generation task.
    """
    result = await db.execute(
        select(GenerationTask).where(
            GenerationTask.task_id == task_id,
            GenerationTask.user_id == current_user.id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    if task.status == TaskStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=task.error_message or "Generation failed"
        )
    
    if task.status != TaskStatus.COMPLETED:
        return GenerationProgress(
            task_id=task_id,
            status=task.status,
            progress=task.progress,
            message="Generation in progress"
        )
    
    return GenerationResponse(
        task_id=task_id,
        success=True,
        audio_url=task.output_path,
        duration=task.output_metadata.get("duration", 0),
        sample_rate=task.output_metadata.get("sample_rate", 44100),
        metadata=task.output_metadata
    )


@router.get("/tasks", response_model=list[GenerationProgress])
async def list_generation_tasks(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    List all generation tasks for the current user.
    """
    result = await db.execute(
        select(GenerationTask)
        .where(GenerationTask.user_id == current_user.id)
        .order_by(GenerationTask.created_at.desc())
        .limit(limit)
    )
    tasks = result.scalars().all()
    
    return [
        GenerationProgress(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            message=task.input_params.get("prompt", "")[:50] + "..."
        )
        for task in tasks
    ]

