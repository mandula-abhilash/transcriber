import torch
import logging
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class TeluguWhisperService:
    def __init__(self, model_name: str = "vasista22/whisper-telugu-large-v2"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.initialize_whisper()

    def initialize_whisper(self) -> None:
        """Initialize Whisper model and processor with proper error handling"""
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            
            logger.info(f"Initializing Whisper Telugu model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name).to(self.device)
            
            # Set the forced decoder IDs for Telugu
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="te",
                task="transcribe"
            )
            self.model.config.forced_decoder_ids = forced_decoder_ids
            self.model.config.suppress_tokens = []
            
            logger.info("Successfully initialized Whisper Telugu components")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper Telugu components: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def validate_audio(self, audio_array: np.ndarray, sampling_rate: int) -> Tuple[bool, str]:
        """Validate audio array before processing"""
        if len(audio_array) == 0:
            return False, "Audio file is empty"
            
        if not np.isfinite(audio_array).all():
            return False, "Audio contains invalid values (inf or nan)"
            
        if len(audio_array.shape) != 1:
            return False, f"Unexpected audio shape: {audio_array.shape}"
            
        min_length = sampling_rate  # 1 second minimum
        if len(audio_array) < min_length:
            return False, f"Audio too short: {len(audio_array)} samples"
            
        return True, ""

    def load_audio(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load and preprocess audio file with validation"""
        try:
            logger.info(f"Loading audio from: {audio_path}")
            
            audio_array, sampling_rate = librosa.load(
                audio_path,
                sr=16000,
                mono=True
            )
            
            logger.info(f"Audio array shape: {audio_array.shape}")
            logger.info(f"Sample rate: {sampling_rate}")
            
            # Validate the loaded audio
            is_valid, error_message = self.validate_audio(audio_array, sampling_rate)
            if not is_valid:
                logger.error(error_message)
                return None
            
            # Normalize audio
            audio_array = librosa.util.normalize(audio_array)
            if not np.isfinite(audio_array).all():
                logger.error("Normalized audio contains invalid values")
                return None
            
            # Reshape for model input
            audio_array = audio_array.reshape(1, -1)
            logger.info(f"Final array shape: {audio_array.shape}")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return None

    def transcribe(self, audio_path: Union[str, Path]) -> Optional[str]:
        """Transcribe Telugu audio with comprehensive error handling"""
        try:
            # Load and preprocess audio
            audio_array = self.load_audio(str(audio_path))
            if audio_array is None:
                return None
            
            # Process audio with processor
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            logger.info("Generating transcription")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features=inputs.input_features,
                    max_new_tokens=448,
                    num_beams=4,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    temperature=0.7
                )
            
            # Decode the generated ids
            transcribed_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Validate transcription result
            if not transcribed_text or not transcribed_text[0].strip():
                logger.error("Empty transcription result")
                return None
            
            result = transcribed_text[0].strip()
            logger.info(f"Successfully transcribed text length: {len(result)}")
            logger.info("First 100 characters: " + result[:100] + "...")
            
            return result
            
        except IndexError as e:
            logger.error(f"Index error during transcription: {str(e)}")
            logger.error(f"Generated IDs shape: {generated_ids.shape if 'generated_ids' in locals() else 'Not available'}")
            return None
            
        except Exception as e:
            logger.error(f"Error during Telugu transcription: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception args: {e.args}")
            return None