from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "")
    USE_WHISPER_TELUGU_LARGE_V2: bool = os.getenv("USE_WHISPER_TELUGU_LARGE_V2", "false").lower() == "true"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()