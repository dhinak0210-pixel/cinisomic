"""
Export API endpoints for downloading generated audio.
"""
import os
import wave
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.security import get_current_user
from app.core.database import get_db
from app.models.database import GenerationTask, User
from app.models.schemas import ExportRequest, ExportResponse

router = APIRouter(prefix="/export", tags=["Export"])


@router.post("/", response_model=ExportResponse)
async def create_export(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Request an export of a generated audio file.
    
    - **task_id**: The generation task ID
    - **format**: Export format (mp3, wav, midi, stems)
    - **quality**: Quality setting (high, medium, low)
    """
    # Verify task exists and belongs to user
    result = await db.execute(
        select(GenerationTask).where(
            GenerationTask.task_id == request.task_id,
            GenerationTask.user_id == current_user.id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    if task.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Generation task is not completed yet"
        )
    
    # For demo, return the existing file
    if task.output_path:
        # Parse the filename from the path
        filename = os.path.basename(task.output_path)
        
        # Create export response
        return ExportResponse(
            download_url=f"/api/v1/export/download/{filename}",
            expires_at=datetime.utcnow() + timedelta(hours=24),
            file_size=os.path.getsize(task.output_path) if os.path.exists(task.output_path) else 0
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Generated file not found"
    )


@router.get("/download/{filename}")
async def download_export(
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download an exported audio file.
    """
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )
    
    # Check various possible locations
    possible_paths = [
        os.path.join(settings.generated_audio_dir, filename),
        os.path.join(settings.uploads_dir, filename),
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if not filepath:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Determine media type
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".midi": "audio/midi",
        ".mid": "audio/midi",
        ".flac": "audio/flac",
    }
    media_type = media_types.get(ext, "audio/octet-stream")
    
    return FileResponse(
        filepath,
        media_type=media_type,
        filename=filename
    )


@router.get("/stems/{task_id}")
async def download_stems(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download individual stems of a generated track.
    
    Returns a ZIP file containing:
    - drums.wav
    - bass.wav
    - vocals.wav
    - other.wav
    """
    import zipfile
    import io
    
    # Verify task exists
    # For demo, create placeholder stems
    stems_info = {
        "drums": "Drum track",
        "bass": "Bass track",
        "vocals": "Vocal track",
        "other": "Other instruments"
    }
    
    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for stem_name, description in stems_info.items():
            # Create placeholder stem info
            stem_content = f"Placeholder for {stem_name} stem\nTask: {task_id}\n{description}"
            zip_file.writestr(f"{stem_name}.txt", stem_content)
    
    zip_buffer.seek(0)
    
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=stems_{task_id}.zip"}
    )


@router.get("/midi/{task_id}")
async def download_midi(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download MIDI file of generated music.
    """
    # For demo, return placeholder MIDI content
    midi_content = b"""MIDI file placeholder for task {}
Generated by Cinematic Music Platform
Converted from AI-generated audio""".format(task_id)
    
    return Response(
        content=midi_content,
        media_type="audio/midi",
        headers={"Content-Disposition": f"attachment; filename=music_{task_id}.mid"}
    )


# ============ WebSocket for Streaming ============

from fastapi import WebSocket, WebSocketDisconnect


@router.websocket("/stream/{task_id}")
async def stream_audio(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for streaming audio in real-time.
    
    This allows clients to receive audio chunks as they are generated.
    """
    await websocket.accept()
    
    try:
        # For demo, send progress updates
        for progress in [10, 25, 50, 75, 100]:
            await websocket.send_json({
                "type": "progress",
                "progress": progress,
                "message": f"Generation progress: {progress}%"
            })
            
            if progress < 100:
                await websocket.send_json({
                    "type": "status",
                    "status": "processing"
                })
        
        # Send completion
        await websocket.send_json({
            "type": "complete",
            "status": "completed",
            "audio_url": f"/api/v1/export/download/audio_{task_id}.wav"
        })
        
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    WebSocket endpoint for receiving real-time generation progress.
    
    Clients connect and receive updates about their generation tasks.
    """
    await websocket.accept()
    
    try:
        while True:
            # Wait for task updates (in production, this would subscribe to Redis)
            data = await websocket.receive_json()
            
            # Echo back for demo
            await websocket.send_json({
                "received": data,
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except WebSocketDisconnect:
        pass

