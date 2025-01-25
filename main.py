import openai
import os
import yt_dlp
import subprocess
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import re

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client(api_key=api_key)

# Set FFmpeg path in environment variables for Windows/Linux compatibility
ffmpeg_path = r"D:\Others\Software\ffmpeg\ffmpeg-2025-01-22-git-e20ee9f9ae-full_build\bin"
if os.name == "nt":  # Windows
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Function to create a date-based folder and return the path
def get_save_directory():
    today = datetime.now().strftime("%Y-%m-%d")
    save_dir = Path("files") / today
    save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    return save_dir

# Function to sanitize filenames based on requirements
def sanitize_filename(title):
    title = re.sub(r'[^\w\s-]', '', title)  # Remove special characters except spaces and hyphens
    title = re.sub(r'\s+', ' ', title).strip()  # Remove multiple spaces and trim spaces
    title = title.replace(' ', '-')  # Replace spaces with hyphens
    return title.lower()  # Convert to lowercase

# Function to get video title and construct save path
def get_video_info(url):
    try:
        ydl_opts = {'quiet': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_title = info.get('title', 'untitled')
            sanitized_title = sanitize_filename(raw_title)
            return sanitized_title
    except Exception as e:
        print(f"Failed to get video title: {e}")
        return None

# Function to download audio and return the final file path
def download_audio_from_youtube(url):
    save_dir = get_save_directory()
    title = get_video_info(url)
    timestamp = datetime.now().strftime("%H-%M-%S")

    if title:
        filename = f"{title}-{timestamp}.mp3"
    else:
        filename = f"{timestamp}.mp3"

    output_file = save_dir / filename

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_file.with_suffix("")),  # Remove .mp3 to avoid duplication
        'quiet': False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Audio downloaded successfully: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Function to re-encode audio to ensure compatibility
def reencode_audio(input_file):
    output_file = input_file.parent / f"processed_{input_file.name}"
    try:
        command = [
            "ffmpeg",
            "-y",  # Overwrite existing file
            "-i", str(input_file),
            "-acodec", "libmp3lame",
            "-ar", "16000",  # Set sample rate to 16kHz
            "-b:a", "192k",
            str(output_file)
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Audio re-encoded successfully.")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding audio: {e}")
        return None

# Paste the YouTube URL here
# url = "https://www.youtube.com/watch?v=Chz4xhji6xs"
url = "https://www.youtube.com/watch?v=e3WcPlMUxGA"

# Download and process the audio
audio_path = download_audio_from_youtube(url)

if audio_path and audio_path.exists():
    processed_audio = reencode_audio(audio_path)
    if processed_audio and processed_audio.exists():
        with open(processed_audio, "rb") as audio_file:
            try:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                print("\nTranscription:\n")
                print(transcript.text)
            except openai.OpenAIError as e:
                print(f"Error during transcription: {e}")
    else:
        print("Error processing audio file after re-encoding.")
else:
    print("Error processing audio file.")
