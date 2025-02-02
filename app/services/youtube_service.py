from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import yt_dlp
import openai
import os
import subprocess
from datetime import datetime
from pathlib import Path
import re
from ..config import settings
import numpy as np
import librosa
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with empty key - will be updated from request
client = openai.OpenAI(api_key="")

def update_api_key(api_key: str):
    """Update the OpenAI client with a new API key"""
    global client
    client = openai.OpenAI(api_key=api_key)

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

def get_proxy_config():
    """Get Bright Data proxy authentication"""
    if not settings.LUMINATI_USERNAME or not settings.LUMINATI_PASSWORD:
        return None
    
    proxy_url = f"http://{settings.LUMINATI_USERNAME}:{settings.LUMINATI_PASSWORD}@brd.superproxy.io:33335"
    return {
        'http': proxy_url,
        'https': proxy_url
    }

# Initialize the model and processor if using Whisper Telugu Large v2
if settings.USE_WHISPER_TELUGU_LARGE_V2:
    logger.info("Initializing Whisper Telugu Large v2 model and processor")
    model_name = "vasista22/whisper-telugu-large-v2"
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)
        model.config.no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        logger.info("Successfully initialized Whisper Telugu Large v2 model and processor")
    except Exception as e:
        logger.error(f"Failed to initialize Whisper Telugu Large v2 model: {e}")
        raise

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

def load_audio_for_telugu(audio_path):
    try:
        logger.info(f"Loading audio from: {audio_path}")
        
        audio_array, sampling_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )
        
        logger.info(f"Audio array shape: {audio_array.shape}")
        logger.info(f"Sample rate: {sampling_rate}")
        
        if len(audio_array) == 0:
            logger.error("Audio file is empty")
            return None
            
        if not np.isfinite(audio_array).all():
            logger.warning("Audio contains invalid values (inf or nan)")
            audio_array = np.nan_to_num(audio_array)
        
        audio_array = librosa.util.normalize(audio_array)
        if not np.isfinite(audio_array).all():
            logger.error("Normalized audio contains invalid values")
            return None
        
        if len(audio_array.shape) != 1:
            logger.error(f"Unexpected audio shape: {audio_array.shape}")
            return None
            
        min_length = 16000
        if len(audio_array) < min_length:
            logger.error(f"Audio too short: {len(audio_array)} samples")
            return None
            
        audio_array = audio_array.reshape(1, -1)
        logger.info(f"Final array shape: {audio_array.shape}")
        
        return audio_array
    except Exception as e:
        logger.error(f"Error loading audio for Telugu transcription: {e}")
        return None

def transcribe_audio(audio_path, source_language=None):
    if not audio_path or not Path(audio_path).exists():
        logger.error("Audio file does not exist")
        return None, None
        
    try:
        iso_language = get_iso_language_code(source_language)

        # Use Hugging Face model for Telugu transcription if flag is enabled
        if source_language == "telugu" and settings.USE_WHISPER_TELUGU_LARGE_V2:
            logger.info("Starting Telugu transcription with Whisper Telugu Large v2")
            try:
                logger.info("Loading and preprocessing audio")
                audio_array = load_audio_for_telugu(audio_path)
                if audio_array is None:
                    logger.error("Failed to load audio for Telugu transcription")
                    return None, None
                
                logger.info(f"Input features shape before processing: {audio_array.shape}")
                
                logger.info("Processing audio with Whisper processor")
                inputs = processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                
                if "input_features" not in inputs:
                    logger.error("Processor failed to generate input features")
                    return None, None
                    
                logger.info(f"Processed input features shape: {inputs['input_features'].shape}")
                
                logger.info("Getting decoder prompt IDs for Telugu")
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="te", task="transcribe")
                
                try:
                    logger.info("Generating transcription with model")
                    with torch.no_grad():
                        predicted_ids = model.generate(
                            inputs["input_features"],
                            max_length=448,
                            no_repeat_ngram_size=3,
                            length_penalty=2.0,
                            num_beams=4,
                            forced_decoder_ids=forced_decoder_ids,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        
                    if hasattr(predicted_ids, 'sequences'):
                        sequences = predicted_ids.sequences
                        logger.info(f"Generated sequences shape: {sequences.shape}")
                    else:
                        logger.error("No sequences in generated output")
                        return None, None

                except RuntimeError as e:
                    logger.error(f"Error during model generation: {e}")
                    logger.error(f"Model device: {next(model.parameters()).device}")
                    logger.error(f"Input device: {inputs['input_features'].device}")
                    return None, None
                
                try:
                    logger.info("Decoding transcribed text")
                    transcribed_text = processor.batch_decode(
                        sequences,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )[0].strip()
                    
                    if not transcribed_text:
                        logger.error("Empty transcription result")
                        return None, None
                    
                    logger.info(f"Successfully transcribed text length: {len(transcribed_text)}")
                    logger.info("First 100 characters of transcription: " + transcribed_text[:100] + "...")
                    return transcribed_text, "te"
                    
                except Exception as e:
                    logger.error(f"Error during text decoding: {e}")
                    return None, None
               
            except Exception as e:
                logger.error(f"Error during Telugu transcription: {e}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Exception args: {e.args}")
                return None, None

        # Use OpenAI Whisper for other languages or Telugu when flag is disabled
        with open(audio_path, "rb") as audio_file:
            # For Telugu when not using the specialized model, don't specify language
            use_language = None if source_language == "telugu" else iso_language
            
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=use_language,
                response_format="verbose_json"
            )
            
            transcribed_text = transcript.text
            detected_language = transcript.language

            # If Telugu and not using Whisper Telugu Large v2, translate using GPT-4
            if source_language == 'telugu' and not settings.USE_WHISPER_TELUGU_LARGE_V2:
                translated_text = translate_telugu_to_english(transcribed_text)
                return translated_text, "te"  # Return "te" as language since we know it's Telugu

            return transcribed_text, detected_language

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None, None

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