from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import logging
import librosa
import numpy as np
from pathlib import Path
from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = None
processor = None

def initialize_whisper():
    """Initialize Whisper model and processor"""
    global model, processor
    
    if settings.USE_WHISPER_TELUGU_LARGE_V2:
        logger.info("Initializing Whisper Telugu Large v2 model and processor")
        try:
            model_name = "vasista22/whisper-telugu-large-v2"
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
            
            # Set the forced decoder IDs for Telugu in the model config
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="te",
                task="transcribe"
            )
            model.config.forced_decoder_ids = forced_decoder_ids
            model.config.suppress_tokens = []

            logger.info("Successfully initialized Whisper Telugu model and processor")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper Telugu components: {e}")
            raise

def load_audio_for_telugu(audio_path):
    """Enhanced audio loading function with preprocessing for Telugu transcription"""
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

def transcribe_telugu(audio_path):
    """Transcribe Telugu audio using Whisper Telugu Large v2"""
    try:
        # Use the enhanced Telugu audio loading function
        audio_array = load_audio_for_telugu(str(audio_path))
        if audio_array is None:
            logger.error("Failed to load audio")
            return None
        
        # Process audio with processor
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        # Generate transcription
        logger.info("Generating transcription")
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=inputs.input_features,
                max_new_tokens=448,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                temperature=0.7
            )
        
        # Decode the generated ids
        transcribed_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        if not transcribed_text:
            logger.error("Empty transcription result")
            return None
        
        logger.info(f"Successfully transcribed text length: {len(transcribed_text)}")
        logger.info("First 100 characters of transcription: " + transcribed_text[:100] + "...")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Error during Telugu transcription: {e}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        return None

# Initialize Whisper components on module load
initialize_whisper()