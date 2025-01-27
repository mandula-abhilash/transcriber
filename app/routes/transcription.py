from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from ..services import youtube_service
from typing import Optional

router = APIRouter()

class TranscriptionRequest(BaseModel):
    url: HttpUrl

class TranscriptionResponse(BaseModel):
    text: str
    audio_path: str

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(request: TranscriptionRequest):
    try:
        audio_path = youtube_service.download_audio_from_youtube(str(request.url))
        if not audio_path:
            raise HTTPException(status_code=400, detail="Failed to download audio")
        
        processed_audio = youtube_service.reencode_audio(audio_path)
        if not processed_audio:
            raise HTTPException(status_code=400, detail="Failed to process audio")
        
        transcript = youtube_service.transcribe_audio(processed_audio)
        if not transcript:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
        
        return TranscriptionResponse(
            text=transcript,
            audio_path=str(processed_audio)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))