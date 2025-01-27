from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import transcription
from .config import settings

app = FastAPI(
    title="YouTube Transcriber API",
    description="API for downloading and transcribing YouTube videos",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcription.router, prefix="/api/v1", tags=["transcription"])