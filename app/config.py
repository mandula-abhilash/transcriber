from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "")
    USE_WHISPER_TELUGU_LARGE_V2: bool = os.getenv("USE_WHISPER_TELUGU_LARGE_V2", "false").lower() == "true"
    
    # Single API URL configuration
    API_URL: str = os.getenv("API_URL", "")
    
    # File storage configuration
    FILES_BASE_DIR: str = os.getenv("FILES_BASE_DIR", "files")
    
    # YouTube cookies path
    YOUTUBE_COOKIES_PATH: str = os.getenv("YOUTUBE_COOKIES_PATH", "")
    
    # Bright Data proxy configuration
    LUMINATI_USERNAME: str = os.getenv("LUMINATI_USERNAME", "")
    LUMINATI_PASSWORD: str = os.getenv("LUMINATI_PASSWORD", "")
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()