import openai
import os
import yt_dlp
import subprocess
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client(api_key=api_key)

# Set FFmpeg path in environment variables
ffmpeg_path = r"D:\Others\Software\ffmpeg\ffmpeg-2025-01-22-git-e20ee9f9ae-full_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Function to download audio from YouTube
def download_audio_from_youtube(url, save_path="downloaded_audio"):
    ydl_opts = {
        'format': 'bestaudio/best',  # Download the best audio quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': save_path,  # Remove .mp3 to avoid double extension issue
        'quiet': False  # Set to True to suppress logs
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        final_path = f"{save_path}.mp3"
        print(f"Audio downloaded successfully: {final_path}")
        return final_path
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Function to re-encode audio to ensure compatibility
def reencode_audio(input_file, output_file="processed_audio.mp3"):
    try:
        command = [
            "ffmpeg",
            "-y",  # Overwrite existing file
            "-i", input_file,  # Input file
            "-acodec", "libmp3lame",  # Re-encode using MP3 codec
            "-ar", "16000",  # Set sample rate to 16kHz (recommended by OpenAI)
            "-b:a", "192k",  # Set bitrate
            output_file
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Audio re-encoded successfully.")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding audio: {e}")
        return None

# Paste the YouTube URL here
url = "https://www.youtube.com/watch?v=Chz4xhji6xs"

# Download and process the audio
audio_path = download_audio_from_youtube(url)

if audio_path and os.path.exists(audio_path):
    processed_audio = reencode_audio(audio_path)
    if processed_audio and os.path.exists(processed_audio):
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
