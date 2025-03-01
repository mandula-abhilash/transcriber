from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from ..services import youtube_service
from typing import Optional
from ..config import settings

router = APIRouter()

class TranscriptionRequest(BaseModel):
    url: HttpUrl
    source_language: Optional[str] = None
    api_key: str

class TranscriptionResponse(BaseModel):
    original_text: str
    detected_language: str
    english_text: Optional[str]
    audio_path: str

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(request: TranscriptionRequest):
    try:
        # Update OpenAI API key from request
        youtube_service.update_api_key(request.api_key)
        
        audio_path = youtube_service.download_audio_from_youtube(str(request.url))
        if not audio_path:
            raise HTTPException(status_code=400, detail="Failed to download audio")
        
        processed_audio = youtube_service.reencode_audio(audio_path)
        if not processed_audio:
            raise HTTPException(status_code=400, detail="Failed to process audio")
        
        transcript, detected_language = youtube_service.transcribe_audio(
            processed_audio, 
            source_language=request.source_language
        )
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
        
        english_text = None
        if detected_language != "en":
            english_text = youtube_service.translate_to_english(transcript, detected_language)
        
        return TranscriptionResponse(
            original_text=transcript,
            detected_language=detected_language,
            english_text=english_text,
            audio_path=str(processed_audio)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))