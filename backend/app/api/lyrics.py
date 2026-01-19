"""
Lyrics generation API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import get_current_user
from app.models.database import LyricsProject, User
from app.models.schemas import (
    LyricsGenerationRequest,
    LyricsProjectResponse,
    LyricsSyncRequest
)
from app.core.database import get_db

router = APIRouter(prefix="/lyrics", tags=["Lyrics Generation"])


# Demo lyrics for testing
DEMO_LYRICS = {
    "epic": """In the shadow of ancient mountains,
We stand united, hearts ablaze,
The drums of war begin to thunder,
Through endless nights and golden days.
With courage burning bright within us,
We'll face the storm and see it through,
For glory waits beyond the horizon,
Where the brave will find their truth.""",
    
    "dramatic": """The weight of silence fills the room,
Shadows dance upon the wall,
Every breath we take is borrowed,
Every step could be our fall.
In the darkness, there is power,
In the silence, there is peace,
But the storm is always waiting,
For the moment of release.""",
    
    "peaceful": """Soft as morning light through leaves,
Gentle as the flowing stream,
Peace within the quiet forest,
Where the world is still a dream.
Birds are singing sweet melodies,
Sunset paints the sky in gold,
In this moment, time stands still,
And the heart is calm and bold.""",
    
    "romantic": """Your eyes hold the depth of oceans,
Your voice is the summer rain,
In your arms I find my home again,
Away from all the world`s pain.
Every kiss is poetry written,
Every touch is music played,
Forever and beyond the horizon,
Together we will stay.""",
    
    "tense": """Something`s watching in the darkness,
Footsteps echo in the hall,
Every shadow holds a secret,
Every breath could break the call.
The walls are closing in around us,
The clock is ticking down the night,
Survival is the only answer,
We must run before the light."""
}


@router.post("/generate")
async def generate_lyrics(
    request: LyricsGenerationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate lyrics based on a prompt.
    
    - **prompt**: Description of the lyrics to generate
    - **style**: Style of lyrics (cinematic, pop, rock, etc.)
    - **mood**: Emotional tone
    - **language**: Language code
    - **length**: Approximate number of words
    """
    # For demo purposes, select lyrics based on mood
    mood = request.mood.lower()
    lyrics_text = DEMO_LYRICS.get(mood, DEMO_LYRICS["epic"])
    
    # Create lyrics project
    project = LyricsProject(
        user_id=current_user.id,
        title=f"Generated Lyrics - {request.mood}",
        lyrics=lyrics_text,
        language=request.language,
        generation_params=request.model_dump()
    )
    
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    return {
        "id": project.id,
        "title": project.title,
        "lyrics": project.lyrics,
        "language": project.language,
        "message": "Lyrics generated successfully"
    }


@router.get("/", response_model=list[LyricsProjectResponse])
async def list_lyrics_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all lyrics projects for the current user.
    """
    result = await db.execute(
        select(LyricsProject)
        .where(LyricsProject.user_id == current_user.id)
        .order_by(LyricsProject.created_at.desc())
    )
    projects = result.scalars().all()
    return projects


@router.get("/{project_id}", response_model=LyricsProjectResponse)
async def get_lyrics_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific lyrics project.
    """
    result = await db.execute(
        select(LyricsProject).where(
            LyricsProject.id == project_id,
            LyricsProject.user_id == current_user.id
        )
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lyrics project not found"
        )
    
    return project


@router.post("/{project_id}/sync", response_model=dict)
async def sync_lyrics(
    project_id: int,
    request: LyricsSyncRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Synchronize lyrics with timestamps.
    
    - **sync_data**: List of lines with start and end times
    """
    result = await db.execute(
        select(LyricsProject).where(
            LyricsProject.id == project_id,
            LyricsProject.user_id == current_user.id
        )
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lyrics project not found"
        )
    
    # Update sync data
    project.sync_data = request.sync_data
    await db.commit()
    
    return {
        "message": "Lyrics synchronized successfully",
        "total_lines": len(request.sync_data)
    }


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lyrics_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a lyrics project.
    """
    result = await db.execute(
        select(LyricsProject).where(
            LyricsProject.id == project_id,
            LyricsProject.user_id == current_user.id
        )
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lyrics project not found"
        )
    
    await db.delete(project)
    await db.commit()

