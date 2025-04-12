"""
TTS Engine interfaces and implementations.

This module provides a common interface for different Text-to-Speech engines
and implements adapters for popular TTS systems.
"""

import os
import abc
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class TTSEngine(abc.ABC):
    """Abstract base class for TTS engines."""
    
    @abc.abstractmethod
    def generate_speech(self, text: str, options: Dict[str, Any] = None) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: The text to synthesize
            options: Engine-specific options for speech generation
            
        Returns:
            Tuple containing audio array and sample rate
        """
        pass
    
    @abc.abstractmethod
    def save_to_file(self, audio_data: Tuple[np.ndarray, int], output_path: str) -> str:
        """
        Save the generated audio to a file.
        
        Args:
            audio_data: Tuple containing audio array and sample rate
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved audio file
        """
        pass
    
    def process_ssml(self, ssml_text: str) -> str:
        """
        Process SSML markup if the engine supports it.
        Default implementation just strips SSML tags.
        
        Args:
            ssml_text: Text with SSML markup
            
        Returns:
            Processed text ready for the TTS engine
        """
        # Basic SSML tag removal - engines should override this if they support SSML
        import re
        return re.sub(r'<[^>]+>', '', ssml_text)
    
    @property
    @abc.abstractmethod
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass
    
    @property
    @abc.abstractmethod
    def supports_emotions(self) -> bool:
        """Whether this engine supports emotional speech."""
        pass
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the TTS engine."""
        pass


class BarkTTSEngine(TTSEngine):
    """
    Implementation of the Bark TTS engine from Suno.
    
    Bark is a transformer-based text-to-audio model that can generate
    highly natural speech with various voices and emotion.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the Bark TTS engine.
        
        Args:
            model_path: Optional path to model files
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = device
        self.model_path = model_path
        self.loaded = False
        self._voices = {}
        
        try:
            # Defer imports to avoid loading unnecessary dependencies
            import torch
            from transformers import AutoProcessor, BarkModel
            
            self.torch = torch
            self.processor_cls = AutoProcessor
            self.model_cls = BarkModel
            
            # Check CUDA availability if requested
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
                
            logger.info(f"Initializing Bark TTS engine (device: {self.device})")
            
        except ImportError as e:
            logger.error(f"Failed to import Bark dependencies: {e}")
            logger.error("Please install with: pip install transformers bark")
            raise ImportError("Bark dependencies not installed") from e
    
    def _ensure_loaded(self):
        """Ensure the model is loaded."""
        if not self.loaded:
            try:
                logger.info("Loading Bark TTS model...")
                self.processor = self.processor_cls.from_pretrained("suno/bark", cache_dir=self.model_path)
                self.model = self.model_cls.from_pretrained("suno/bark", cache_dir=self.model_path)
                self.model.to(self.device)
                self.loaded = True
                logger.info("Bark TTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Bark model: {e}")
                raise RuntimeError("Failed to load Bark model") from e
    
    def generate_speech(self, text: str, options: Dict[str, Any] = None) -> Tuple[np.ndarray, int]:
        """
        Generate speech using Bark TTS.
        
        Args:
            text: Text to synthesize
            options: Options including:
                - voice_preset: Name of voice preset to use
                - emotion: Emotion to convey (e.g., "happy", "sad")
                - temperature: Sampling temperature (higher = more random)
                
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._ensure_loaded()
        
        options = options or {}
        voice_preset = options.get("voice_preset", "v2/en_speaker_6")
        temperature = options.get("temperature", 0.7)
        
        # Add emotion markers if specified
        if options.get("emotion"):
            emotion = options["emotion"].lower()
            emotion_markers = {
                "happy": "[happy]",
                "sad": "[sad]",
                "excited": "[excited]",
                "disappointed": "[disappointed]",
                "questioning": "[questioning]",
                "emphatic": "[emphatic]"
            }
            if emotion in emotion_markers:
                text = f"{emotion_markers[emotion]} {text}"
        
        # Process text - add proper spacing and breaks
        text = self._process_text_for_bark(text)
        
        try:
            # Generate speech
            inputs = self.processor(
                text=text,
                voice_preset=voice_preset,
                return_tensors="pt"
            ).to(self.device)
            
            with self.torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature
                )
            
            # Convert to numpy array
            audio_array = output.cpu().numpy().squeeze()
            sample_rate = self.model.generation_config.sample_rate
            
            return audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Error generating speech with Bark: {e}")
            raise RuntimeError(f"Failed to generate speech: {e}") from e
    
    def _process_text_for_bark(self, text: str) -> str:
        """Process text to make it more suitable for Bark TTS."""
        # Add appropriate pauses and breaks
        processed_text = text.replace("...", "[long_pause]")
        processed_text = processed_text.replace("--", "[pause]")
        
        # Ensure proper spacing for punctuation
        for punct in [".", ",", "!", "?", ";"]:
            processed_text = processed_text.replace(f"{punct} ", f"{punct} ")
        
        return processed_text
    
    def save_to_file(self, audio_data: Tuple[np.ndarray, int], output_path: str) -> str:
        """
        Save the generated audio to a WAV file.
        
        Args:
            audio_data: Tuple of (audio_array, sample_rate)
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved audio file
        """
        from scipy.io import wavfile
        
        audio_array, sample_rate = audio_data
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save audio file
        try:
            wavfile.write(output_path, sample_rate, audio_array)
            logger.info(f"Saved audio to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            raise RuntimeError(f"Failed to save audio: {e}") from e
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return ["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"]
    
    @property
    def supports_emotions(self) -> bool:
        """Whether this engine supports emotional speech."""
        return True
    
    @property
    def name(self) -> str:
        """Return the name of the TTS engine."""
        return "Bark"


class CoquiTTSEngine(TTSEngine):
    """
    Implementation of the Coqui TTS engine.
    
    Coqui TTS is an open-source Text-to-Speech system with various models
    and voices, offering good quality with lower resource requirements than Bark.
    """
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", device: str = "cuda"):
        """
        Initialize the Coqui TTS engine.
        
        Args:
            model_name: Name of the Coqui TTS model to use
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.loaded = False
        
        try:
            # Defer imports to avoid loading unnecessary dependencies
            import torch
            self.torch = torch
            
            # Check CUDA availability if requested
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
                
            logger.info(f"Initializing Coqui TTS engine (device: {self.device}, model: {self.model_name})")
            
        except ImportError as e:
            logger.error(f"Failed to import Coqui TTS dependencies: {e}")
            logger.error("Please install with: pip install TTS")
            raise ImportError("Coqui TTS dependencies not installed") from e
    
    def _ensure_loaded(self):
        """Ensure the model is loaded."""
        if not self.loaded:
            try:
                from TTS.api import TTS
                
                logger.info(f"Loading Coqui TTS model: {self.model_name}...")
                self.tts = TTS(model_name=self.model_name, progress_bar=False)
                
                # Set device
                self.tts.to(self.device)
                self.loaded = True
                logger.info("Coqui TTS model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load Coqui TTS model: {e}")
                raise RuntimeError("Failed to load Coqui TTS model") from e
    
    def generate_speech(self, text: str, options: Dict[str, Any] = None) -> Tuple[np.ndarray, int]:
        """
        Generate speech using Coqui TTS.
        
        Args:
            text: Text to synthesize
            options: Options including:
                - speaker: Speaker ID (for multi-speaker models)
                - language: Language code (for multilingual models)
                
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._ensure_loaded()
        
        options = options or {}
        speaker = options.get("speaker", None)
        language = options.get("language", None)
        
        # Process text for better synthesis
        text = self._process_text_for_tts(text)
        
        try:
            # Generate speech
            wav = self.tts.tts(
                text=text,
                speaker=speaker,
                language=language
            )
            
            # For Coqui TTS, wav is already a numpy array
            # Get sample rate from the model
            sample_rate = self.tts.synthesizer.output_sample_rate
            
            return wav, sample_rate
            
        except Exception as e:
            logger.error(f"Error generating speech with Coqui TTS: {e}")
            raise RuntimeError(f"Failed to generate speech: {e}") from e
    
    def _process_text_for_tts(self, text: str) -> str:
        """Process text to make it more suitable for TTS."""
        # Add appropriate pauses
        processed_text = text.replace("...", ", ")
        processed_text = processed_text.replace("--", ", ")
        
        # Remove multiple spaces
        processed_text = " ".join(processed_text.split())
        
        return processed_text
    
    def save_to_file(self, audio_data: Tuple[np.ndarray, int], output_path: str) -> str:
        """
        Save the generated audio to a WAV file.
        
        Args:
            audio_data: Tuple of (audio_array, sample_rate)
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved audio file
        """
        from scipy.io import wavfile
        
        audio_array, sample_rate = audio_data
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save audio file
        try:
            wavfile.write(output_path, sample_rate, audio_array)
            logger.info(f"Saved audio to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            raise RuntimeError(f"Failed to save audio: {e}") from e
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        # This will depend on the specific model being used
        # Basic models typically support one language, multilingual models support more
        if not self.loaded:
            return ["en"]  # Default assumption
            
        # For multilingual models, the supported languages would be available
        # This is a simplification, actual implementation would depend on the model
        return getattr(self.tts, "languages", ["en"])
    
    @property
    def supports_emotions(self) -> bool:
        """Whether this engine supports emotional speech."""
        # Most Coqui models do not directly support emotions
        return False
    
    @property
    def name(self) -> str:
        """Return the name of the TTS engine."""
        return f"Coqui-TTS ({self.model_name})"


# Factory function to create TTS engines
def create_tts_engine(engine_type: str, **kwargs) -> TTSEngine:
    """
    Factory function to create a TTS engine.
    
    Args:
        engine_type: Type of TTS engine to create
        **kwargs: Additional arguments to pass to the engine constructor
        
    Returns:
        Initialized TTS engine
    """
    engines = {
        "bark": BarkTTSEngine,
        "coqui": CoquiTTSEngine,
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown TTS engine type: {engine_type}. Available types: {list(engines.keys())}")
    
    return engines[engine_type](**kwargs) 