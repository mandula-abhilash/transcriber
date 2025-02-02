from transformers import pipeline
import yt_dlp
import openai
import os
import subprocess
from datetime import datetime
from pathlib import Path
import re
from ..config import settings
import logging
from . import whisper_service

telugu_transcriber = whisper_service.TeluguWhisperService()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with empty key - will be updated from request
client = openai.OpenAI(api_key="")

# Language code mapping
LANGUAGE_CODES = {
    'hindi': 'hi',
    'telugu': 'te',
    'tamil': 'ta',
    'kannada': 'kn',
    'malayalam': 'ml',
    'bengali': 'bn',
    'marathi': 'mr',
    'gujarati': 'gu',
    'urdu': 'ur',
    'english': 'en',
}

def update_api_key(api_key: str):
    """Update the OpenAI client with a new API key"""
    global client
    client = openai.OpenAI(api_key=api_key)

def get_proxy_config():
    """Get Bright Data proxy authentication"""
    if not settings.LUMINATI_USERNAME or not settings.LUMINATI_PASSWORD:
        return None
    
    proxy_url = f"http://{settings.LUMINATI_USERNAME}:{settings.LUMINATI_PASSWORD}@brd.superproxy.io:33335"
    return {
        'http': proxy_url,
        'https': proxy_url
    }

def get_iso_language_code(language):
    if not language:
        return None
    
    if len(language) == 2 and language.isalpha():
        return language.lower()
    
    language_lower = language.lower()
    return LANGUAGE_CODES.get(language_lower)

if settings.FFMPEG_PATH and os.name == "nt":
    os.environ["PATH"] += os.pathsep + settings.FFMPEG_PATH

def get_save_directory():
    today = datetime.now().strftime("%Y-%m-%d")
    save_dir = Path(settings.FILES_BASE_DIR) / today
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def sanitize_filename(title):
    title = re.sub(r'[^\w\s-]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    title = title.replace(' ', '-')
    return title.lower()

def get_video_info(url):
    try:
        ydl_opts = {
            'quiet': True,
            'noplaylist': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': False,
            'ignoreerrors': True,
            'nocheckcertificate': True,
            'geo_bypass': True,
        }
        
        if settings.YOUTUBE_COOKIES_PATH:
            ydl_opts['cookiefile'] = settings.YOUTUBE_COOKIES_PATH
    
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_title = info.get('title', 'untitled')
            return sanitize_filename(raw_title)
    except Exception as e:
        logger.error(f"Failed to get video title: {e}")
        return None

def download_audio_from_youtube(url):
    save_dir = get_save_directory()
    title = get_video_info(url)
    timestamp = datetime.now().strftime("%H-%M-%S")

    filename = f"{title}-{timestamp}.mp3" if title else f"{timestamp}.mp3"
    output_file = save_dir / filename

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_file.with_suffix("")),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'extract_flat': False,
        'force_generic_extractor': False,
    }
    
    if settings.YOUTUBE_COOKIES_PATH:
        ydl_opts['cookiefile'] = settings.YOUTUBE_COOKIES_PATH
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if output_file.exists():
            logger.info(f"Audio downloaded successfully: {output_file}")
            return output_file
        else:
            logger.error("Download completed but file not found")
            return None
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

def reencode_audio(input_file):
    if not input_file or not input_file.exists():
        logger.error("Input file does not exist")
        return None
        
    output_file = input_file.parent / f"processed_{input_file.name}"
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i", str(input_file),
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",  # Force mono channel
            "-b:a", "192k",
            str(output_file)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error re-encoding audio: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during re-encoding: {e}")
        return None

def translate_telugu_to_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following Telugu text to English while preserving technical terms, names, and brands in their original form. Provide only the direct translation without any explanations, comments, or notes."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during Telugu translation: {e}")
        return None

def translate_to_english(text, source_language):
    try:
        language_name = next((name for name, code in LANGUAGE_CODES.items() 
                            if code == source_language.lower()), source_language)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the following {language_name} text to English while preserving technical terms, names, and brands in their original form. Provide only the direct translation without any explanations, comments, or notes."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return None

def transcribe_audio(audio_path, source_language=None):
    if not audio_path or not Path(audio_path).exists():
        logger.error("Audio file does not exist")
        return None, None
        
    try:
        iso_language = get_iso_language_code(source_language)
        
        # Check if it's Telugu and USE_WHISPER_TELUGU_LARGE_V2 is enabled
        is_telugu = source_language and source_language.lower() in ['telugu', 'te']
        if is_telugu and settings.USE_WHISPER_TELUGU_LARGE_V2:
            logger.info("Starting Telugu transcription with Whisper Telugu Large v2")
            transcribed_text = telugu_transcriber.transcribe(audio_path)
            if transcribed_text:
                return transcribed_text, "te"
            return None, None
        
        # For non-Telugu languages or when Telugu model is disabled, use OpenAI Whisper
        if not is_telugu or not settings.USE_WHISPER_TELUGU_LARGE_V2:
            with open(audio_path, "rb") as audio_file:
                # Don't specify language for OpenAI if it's Telugu
                use_language = None if is_telugu else iso_language
                
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=use_language,
                    response_format="verbose_json"
                )
                
                transcribed_text = transcript.text
                detected_language = transcript.language

                # If Telugu and not using Whisper Telugu Large v2, translate using GPT-4
                if is_telugu and not settings.USE_WHISPER_TELUGU_LARGE_V2:
                    translated_text = translate_telugu_to_english(transcribed_text)
                    return translated_text, "te"

                return transcribed_text, detected_language

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None, None