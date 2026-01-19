
"""
Main FastAPI application for Cinematic Music Platform.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db, close_db
from app.api import music, auth, voice, lyrics, projects
from app.api.export import router as export_router

# Create required directories
os.makedirs(settings.uploads_dir, exist_ok=True)
os.makedirs(settings.generated_audio_dir, exist_ok=True)
os.makedirs(settings.voice_samples_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered cinematic music generation platform",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files for generated audio
app.mount(
    "/generated",
    StaticFiles(directory=settings.generated_audio_dir),
    name="generated"
)


# Include API routers
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(music.router, prefix=settings.api_prefix)
app.include_router(voice.router, prefix=settings.api_prefix)
app.include_router(lyrics.router, prefix=settings.api_prefix)
app.include_router(projects.router, prefix=settings.api_prefix)
app.include_router(export_router, prefix=settings.api_prefix)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Cinematic Music Platform API",
        "version": settings.app_version,
        "docs": "/docs"
    }


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}


# Export download endpoint
@app.get("/api/v1/export/download/{filename}")
async def download_file(filename: str):
    """Download a generated audio file."""
    filepath = os.path.join(settings.generated_audio_dir, filename)
    if os.path.exists(filepath):
        return FileResponse(
            filepath,
            media_type="audio/wav",
            filename=filename
        )
    return {"error": "File not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
