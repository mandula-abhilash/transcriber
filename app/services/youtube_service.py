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

# Initialize OpenAI client without proxies
client = openai.OpenAI(
    api_key=settings.OPENAI_API_KEY,
)

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

# Initialize the model and processor
model_name = "vasista22/whisper-telugu-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Configure generation parameters
model.config.no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

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
    save_dir = Path("files") / today
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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_title = info.get('title', 'untitled')
            return sanitize_filename(raw_title)
    except Exception as e:
        print(f"Failed to get video title: {e}")
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

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if output_file.exists():
            print(f"Audio downloaded successfully: {output_file}")
            return output_file
        else:
            print("Download completed but file not found")
            return None
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def reencode_audio(input_file):
    if not input_file or not input_file.exists():
        print("Input file does not exist")
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
            print(f"FFmpeg error: {result.stderr}")
            return None
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding audio: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error during re-encoding: {e}")
        return None

def load_audio_for_telugu(audio_path):
    try:
        # Load audio with correct parameters
        audio_array, sampling_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )
        
        # Ensure audio is not empty
        if len(audio_array) == 0:
            print("Audio file is empty")
            return None
            
        # Normalize audio
        audio_array = librosa.util.normalize(audio_array)
        
        # Reshape to match model's expected input shape (1, num_samples)
        audio_array = audio_array.reshape(1, -1)
        
        return audio_array
    except Exception as e:
        print(f"Error loading audio for Telugu transcription: {e}")
        return None

def transcribe_audio(audio_path, source_language=None):
    if not audio_path or not Path(audio_path).exists():
        print("Audio file does not exist")
        return None, None
        
    try:
        iso_language = get_iso_language_code(source_language)

        # Use Hugging Face model for Telugu transcription
        if source_language == "telugu":
            try:
                # Load audio as numpy array
                audio_array = load_audio_for_telugu(audio_path)
                if audio_array is None:
                    return None, None
                
                # Process the audio with the feature extractor
                inputs = processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                
                # Generate transcription
                with torch.no_grad():
                    predicted_ids = model.generate(
                        inputs["input_features"],
                        max_length=448,
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        num_beams=4,
                    )
                
                # Decode the output
                transcribed_text = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0].strip()
                
                return transcribed_text, "te"
            except Exception as e:
                print(f"Error during Telugu transcription: {e}")
                return None, None

        # Default to OpenAI Whisper for other languages
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=iso_language if iso_language else None,
                response_format="verbose_json"
            )
            return transcript.text, transcript.language

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None, None

def translate_telugu_to_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following Telugu text to English while preserving technical terms, names, and brands in their original form."
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
        print(f"Error during Telugu translation: {e}")
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
                    "content": f"You are a translator. Translate the following {language_name} text to English while preserving technical terms, names, and brands in their original form. If the text appears to be incorrect or garbled, please indicate that in your response."
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
        print(f"Error during translation: {e}")
        return None