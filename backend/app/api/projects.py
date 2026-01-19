"""
Projects API endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import get_current_user
from app.models.database import Project, Scene, User, Track
from app.models.schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    SceneCreate,
    SceneResponse
)
from app.core.database import get_db

router = APIRouter(prefix="/projects", tags=["Projects"])


# ============ Projects ============

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new project.
    """
    project = Project(
        name=project_data.name,
        description=project_data.description,
        owner_id=current_user.id,
        settings=project_data.settings or {}
    )
    
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    return project


@router.get("/", response_model=list[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """
    List all projects for the current user.
    """
    result = await db.execute(
        select(Project)
        .where(Project.owner_id == current_user.id)
        .order_by(Project.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    projects = result.scalars().all()
    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific project.
    """
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == current_user.id
        )
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    return project


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a project.
    """
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == current_user.id
        )
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Update fields
    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    if project_data.settings is not None:
        project.settings = project_data.settings
    
    await db.commit()
    await db.refresh(project)
    
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a project.
    """
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == current_user.id
        )
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    await db.delete(project)
    await db.commit()


# ============ Scenes ============

@router.get("/{project_id}/scenes", response_model=list[SceneResponse])
async def list_scenes(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all scenes for a project.
    """
    # Verify project ownership
    project = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == current_user.id
        )
    )
    if not project.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    result = await db.execute(
        select(Scene)
        .where(Scene.project_id == project_id)
        .order_by(Scene.start_time)
    )
    scenes = result.scalars().all()
    return scenes


@router.post("/{project_id}/scenes", response_model=SceneResponse, status_code=status.HTTP_201_CREATED)
async def create_scene(
    project_id: int,
    scene_data: SceneCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new scene for a project.
    """
    # Verify project ownership
    project = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == current_user.id
        )
    )
    if not project.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    scene = Scene(
        project_id=project_id,
        name=scene_data.name,
        description=scene_data.description,
        start_time=scene_data.start_time,
        duration=scene_data.duration,
        metadata=scene_data.metadata or {}
    )
    
    db.add(scene)
    await db.commit()
    await db.refresh(scene)
    
    return scene


@router.get("/{project_id}/scenes/{scene_id}", response_model=SceneResponse)
async def get_scene(
    project_id: int,
    scene_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific scene.
    """
    result = await db.execute(
        select(Scene).where(
            Scene.id == scene_id,
            Scene.project_id == project_id
        )
    )
    scene = result.scalar_one_or_none()
    
    if not scene:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found"
        )
    
    return scene


@router.delete("/{project_id}/scenes/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scene(
    project_id: int,
    scene_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a scene.
    """
    result = await db.execute(
        select(Scene).where(
            Scene.id == scene_id,
            Scene.project_id == project_id
        )
    )
    scene = result.scalar_one_or_none()
    
    if not scene:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found"
        )
    
    await db.delete(scene)
    await db.commit()


# ============ Tracks ============

@router.get("/{project_id}/tracks", response_model=list[dict])
async def list_tracks(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all tracks for a project.
    """
    # Verify project ownership
    project = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == current_user.id
        )
    )
    if not project.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    result = await db.execute(
        select(Track)
        .where(Track.project_id == project_id)
        .order_by(Track.created_at)
    )
    tracks = result.scalars().all()
    
    return [
        {
            "id": track.id,
            "name": track.name,
            "track_type": track.track_type,
            "duration": track.duration,
            "created_at": track.created_at.isoformat()
        }
        for track in tracks
    ]

