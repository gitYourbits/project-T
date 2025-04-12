#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Video Teacher - Narration Generator Core Module.

This module provides functionality for generating narration audio from text using
different TTS engines and aligning the generated audio with the original text.

Key features:
- Multiple TTS engines support (Bark, Coqui)
- Multiple text-to-audio alignment methods (Aeneas, Whisper, Gentle)
- Emotion and cadence variation in narration
- Word-level timing information for video generation
"""

import os
import sys
import json
import time
import enum
import logging
import argparse
import tempfile
import importlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import soundfile as sf
import textstat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("narration_generator")

# Define enums for supported TTS engines and aligners
class TTSEngine(enum.Enum):
    BARK = "bark"
    COQUI = "coqui"

class AlignerType(enum.Enum):
    AENEAS = "aeneas"
    WHISPER = "whisper"
    GENTLE = "gentle"

@dataclass
class WordTiming:
    """Word timing information."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0

@dataclass
class NarrationResult:
    """Result of narration generation."""
    audio_path: str
    sample_rate: int
    duration: float
    word_timings: List[WordTiming]
    text: str
    voice_id: str
    tts_engine: TTSEngine
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NarrationConfig:
    """Configuration for narration generation."""
    # TTS engine settings
    engine: TTSEngine = TTSEngine.BARK
    voice_id: str = "v2/en_speaker_6"  # Default Bark voice
    output_dir: str = "output/narration"
    
    # Audio settings
    sample_rate: int = 24000
    
    # Performance settings
    use_cache: bool = True
    cache_dir: str = ".cache/narration"
    
    # Alignment settings
    aligner: AlignerType = AlignerType.AENEAS
    gentle_url: str = "http://localhost:8765/transcriptions"
    
    # Emotion settings
    emotion_enabled: bool = True
    emotion_strength: float = 0.7  # 0.0 to 1.0
    
    # Advanced settings
    temperature: float = 0.7  # Creativity of generation (higher = more creative)
    silence_padding: float = 0.5  # Seconds of silence to add at the beginning/end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enums."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, enum.Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

class NarrationGenerator:
    """
    Core narration generator class for generating audio from text using different TTS engines.
    """
    
    def __init__(self, 
                 tts_engine: Union[TTSEngine, str] = TTSEngine.BARK, 
                 aligner: Union[AlignerType, str] = AlignerType.AENEAS,
                 config: Optional[NarrationConfig] = None,
                 emotion_detection: bool = True):
        """
        Initialize the narration generator.
        
        Args:
            tts_engine: TTS engine to use (TTSEngine enum or string)
            aligner: Aligner to use for word timing (AlignerType enum or string)
            config: Configuration for narration generation
            emotion_detection: Whether to use emotion detection for narration
        """
        # Convert string enums to actual enum values if needed
        if isinstance(tts_engine, str):
            tts_engine = TTSEngine(tts_engine.lower())
            
        if isinstance(aligner, str):
            aligner = AlignerType(aligner.lower())
        
        # Set configuration
        self.config = config or NarrationConfig()
        self.config.engine = tts_engine
        self.config.aligner = aligner
        self.config.emotion_enabled = emotion_detection
        
        # Create output and cache directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.use_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize lazy-loaded components
        self._tts_engine = None
        self._aligner = None
        
        logger.info(f"Initialized NarrationGenerator with {tts_engine.value} TTS engine and {aligner.value} aligner")
    
    def _get_tts_engine(self):
        """Lazy load the TTS engine."""
        if self._tts_engine is not None:
            return self._tts_engine
        
        if self.config.engine == TTSEngine.BARK:
            self._tts_engine = self._init_bark()
        elif self.config.engine == TTSEngine.COQUI:
            self._tts_engine = self._init_coqui()
        else:
            raise ValueError(f"Unsupported TTS engine: {self.config.engine}")
        
        return self._tts_engine
    
    def _get_aligner(self):
        """Lazy load the aligner."""
        if self._aligner is not None:
            return self._aligner
        
        if self.config.aligner == AlignerType.AENEAS:
            self._aligner = self._init_aeneas_aligner()
        elif self.config.aligner == AlignerType.WHISPER:
            self._aligner = self._init_whisper_aligner()
        elif self.config.aligner == AlignerType.GENTLE:
            self._aligner = self._init_gentle_aligner()
        else:
            raise ValueError(f"Unsupported aligner: {self.config.aligner}")
        
        return self._aligner
    
    def _init_bark(self):
        """Initialize the Bark TTS engine."""
        try:
            logger.info("Loading Bark TTS engine...")
            try:
                from bark import SAMPLE_RATE, generate_audio, preload_models
                from bark.generation import generate_text_semantic
                from bark.api import semantic_to_waveform
            except ImportError:
                logger.error("Failed to import Bark. Please install it with: pip install git+https://github.com/suno-ai/bark.git")
                sys.exit(1)
            
            # Preload models
            preload_models()
            logger.info("Bark TTS engine loaded successfully")
            
            return {
                "generate_audio": generate_audio,
                "sample_rate": SAMPLE_RATE,
                "generate_text_semantic": generate_text_semantic,
                "semantic_to_waveform": semantic_to_waveform
            }
        except Exception as e:
            logger.error(f"Error initializing Bark TTS: {e}")
            raise
    
    def _init_coqui(self):
        """Initialize the Coqui TTS engine."""
        try:
            logger.info("Loading Coqui TTS engine...")
            try:
                import TTS
                from TTS.utils.manage import ModelManager
                from TTS.utils.synthesizer import Synthesizer
            except ImportError:
                logger.error("Failed to import Coqui TTS. Please install it with: pip install TTS")
                sys.exit(1)
            
            # Initialize the model manager
            model_manager = ModelManager()
            
            # Get the latest model
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            model_path = model_manager.download_model(model_name)
            
            # Create synthesizer
            synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=None,
                vocoder_checkpoint=None,
                vocoder_config=None,
                use_cuda=False
            )
            
            logger.info("Coqui TTS engine loaded successfully")
            
            return {
                "synthesizer": synthesizer,
                "sample_rate": synthesizer.output_sample_rate
            }
        except Exception as e:
            logger.error(f"Error initializing Coqui TTS: {e}")
            raise
    
    def _init_aeneas_aligner(self):
        """Initialize the Aeneas aligner."""
        try:
            logger.info("Loading Aeneas aligner...")
            try:
                import aeneas
                from aeneas.executetask import ExecuteTask
                from aeneas.task import Task
            except ImportError:
                logger.error("Failed to import Aeneas. Please install it with: pip install aeneas")
                sys.exit(1)
            
            logger.info("Aeneas aligner loaded successfully")
            
            return {
                "ExecuteTask": ExecuteTask,
                "Task": Task
            }
        except Exception as e:
            logger.error(f"Error initializing Aeneas aligner: {e}")
            raise
    
    def _init_whisper_aligner(self):
        """Initialize the Whisper aligner."""
        try:
            logger.info("Loading Whisper aligner...")
            try:
                import whisper
            except ImportError:
                logger.error("Failed to import Whisper. Please install it with: pip install openai-whisper")
                sys.exit(1)
            
            # Load the model (using small model for faster loading)
            model = whisper.load_model("base")
            
            logger.info("Whisper aligner loaded successfully")
            
            return {
                "model": model,
                "align": lambda audio_file, text: self._whisper_align(model, audio_file, text)
            }
        except Exception as e:
            logger.error(f"Error initializing Whisper aligner: {e}")
            raise
    
    def _init_gentle_aligner(self):
        """Initialize the Gentle aligner."""
        try:
            logger.info("Loading Gentle aligner...")
            import requests
            
            # Test connection to Gentle server
            try:
                response = requests.get(self.config.gentle_url.replace("/transcriptions", ""))
                if response.status_code != 200:
                    logger.warning(f"Gentle server at {self.config.gentle_url} might not be running properly")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Could not connect to Gentle server at {self.config.gentle_url}")
                logger.warning("Please ensure the Gentle server is running. See: https://github.com/lowerquality/gentle")
            
            logger.info("Gentle aligner configured successfully")
            
            return {
                "url": self.config.gentle_url,
                "requests": requests
            }
        except Exception as e:
            logger.error(f"Error initializing Gentle aligner: {e}")
            raise
    
    def _generate_bark_audio(self, text: str, voice_id: str, output_path: str) -> Tuple[np.ndarray, int]:
        """Generate audio using Bark TTS."""
        bark = self._get_tts_engine()
        
        # Process the voice_id (can be a preset like "v2/en_speaker_6" or a speaker embedding tensor)
        if isinstance(voice_id, str):
            voice_preset = voice_id
        else:
            voice_preset = voice_id  # Assuming it's already a speaker embedding tensor
        
        # Generate audio
        start_time = time.time()
        audio_array = bark["generate_audio"](text, voice_preset, temperature=self.config.temperature)
        sample_rate = bark["sample_rate"]
        duration = time.time() - start_time
        
        logger.info(f"Generated {len(audio_array) / sample_rate:.2f} seconds of audio in {duration:.2f} seconds")
        
        # Save to file
        sf.write(output_path, audio_array, sample_rate)
        
        return audio_array, sample_rate
    
    def _generate_coqui_audio(self, text: str, voice_id: str, output_path: str) -> Tuple[np.ndarray, int]:
        """Generate audio using Coqui TTS."""
        coqui = self._get_tts_engine()
        synthesizer = coqui["synthesizer"]
        
        # Generate audio
        start_time = time.time()
        # Note: voice_id is ignored for Coqui as we're using a pre-trained model
        wavs = synthesizer.tts(text)
        sample_rate = coqui["sample_rate"]
        duration = time.time() - start_time
        
        logger.info(f"Generated {len(wavs) / sample_rate:.2f} seconds of audio in {duration:.2f} seconds")
        
        # Convert to float32 and normalize
        audio_array = np.array(wavs).astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / audio_array.max()
        
        # Save to file
        sf.write(output_path, audio_array, sample_rate)
        
        return audio_array, sample_rate
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text (simplified version, can be enhanced with ML models)."""
        if not self.config.emotion_enabled:
            return {}
            
        # Very simple emotion detection as a placeholder
        # In a production system, this would use a proper emotion detection model
        emotions = {
            "happy": 0.0,
            "sad": 0.0,
            "excited": 0.0,
            "neutral": 1.0,  # Default to neutral
            "questioning": 0.0
        }
        
        # Simple keyword matching
        happy_words = ["happy", "joy", "excellent", "great", "wonderful", "smile"]
        sad_words = ["sad", "unfortunate", "sorry", "regret", "disappointed"]
        excited_words = ["exciting", "amazing", "fantastic", "incredible", "awesome"]
        question_indicators = ["?", "what", "how", "why", "when", "who", "which"]
        
        text_lower = text.lower()
        
        # Count occurrences
        for word in happy_words:
            if word in text_lower:
                emotions["happy"] += 0.2 * self.config.emotion_strength
                emotions["neutral"] -= 0.1 * self.config.emotion_strength
                
        for word in sad_words:
            if word in text_lower:
                emotions["sad"] += 0.2 * self.config.emotion_strength
                emotions["neutral"] -= 0.1 * self.config.emotion_strength
                
        for word in excited_words:
            if word in text_lower:
                emotions["excited"] += 0.2 * self.config.emotion_strength
                emotions["neutral"] -= 0.1 * self.config.emotion_strength
        
        for indicator in question_indicators:
            if indicator in text_lower:
                emotions["questioning"] += 0.1 * self.config.emotion_strength
                emotions["neutral"] -= 0.05 * self.config.emotion_strength
        
        # Ensure neutral doesn't go below 0
        emotions["neutral"] = max(0.1, emotions["neutral"])
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] /= total
        
        return emotions
    
    def _aeneas_align(self, audio_file: str, text: str) -> List[WordTiming]:
        """Align audio and text using Aeneas."""
        aeneas = self._get_aligner()
        
        # Create a temporary file for the text
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as text_file:
            text_file.write(text)
            text_path = text_file.name
        
        try:
            # Configure the task
            config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"
            task = aeneas["Task"](config_string=config_string)
            task.audio_file_path_absolute = audio_file
            task.text_file_path_absolute = text_path
            
            # Process the alignment
            executetask = aeneas["ExecuteTask"](task)
            executetask.execute()
            
            # Get the alignment
            word_timings = []
            
            # Aeneas doesn't do word-level alignment out of the box, so we split the text
            words = text.split()
            alignment = task.sync_map.fragments
            
            if len(alignment) != len(words):
                logger.warning(f"Aeneas produced {len(alignment)} fragments but text has {len(words)} words")
                # Simple fallback: distribute words evenly across fragments
                for fragment in alignment:
                    begin = fragment.begin
                    end = fragment.end
                    text_fragment = fragment.text
                    
                    # Count words in this fragment
                    fragment_words = text_fragment.split()
                    word_count = len(fragment_words)
                    
                    if word_count == 0:
                        continue
                    
                    # Calculate time per word
                    time_per_word = (end - begin) / word_count
                    
                    # Add each word with its timing
                    for i, word in enumerate(fragment_words):
                        word_begin = begin + i * time_per_word
                        word_end = word_begin + time_per_word
                        
                        word_timings.append(WordTiming(
                            word=word,
                            start_time=word_begin,
                            end_time=word_end,
                            confidence=0.8  # Lower confidence for this fallback method
                        ))
            else:
                # Direct mapping of words to fragments
                for i, fragment in enumerate(alignment):
                    word_timings.append(WordTiming(
                        word=words[i],
                        start_time=fragment.begin,
                        end_time=fragment.end,
                        confidence=0.9
                    ))
            
            return word_timings
        finally:
            # Clean up
            os.unlink(text_path)
    
    def _whisper_align(self, model, audio_file: str, text: str) -> List[WordTiming]:
        """Align audio and text using Whisper."""
        import whisper
        
        # Transcribe the audio
        result = model.transcribe(audio_file)
        
        # Process the segments to get word-level timing
        word_timings = []
        
        if hasattr(result, "segments") and result.segments:
            # Extract words and their timing from segments
            for segment in result.segments:
                # If segment has word-level timestamps
                if hasattr(segment, "words") and segment.words:
                    for word_data in segment.words:
                        word_timings.append(WordTiming(
                            word=word_data["word"],
                            start_time=word_data["start"],
                            end_time=word_data["end"],
                            confidence=word_data.get("confidence", 0.9)
                        ))
                else:
                    # Fallback: split segment text into words and distribute evenly
                    words = segment.text.split()
                    segment_duration = segment.end - segment.start
                    word_duration = segment_duration / len(words) if words else 0
                    
                    for i, word in enumerate(words):
                        word_start = segment.start + i * word_duration
                        word_end = word_start + word_duration
                        
                        word_timings.append(WordTiming(
                            word=word,
                            start_time=word_start,
                            end_time=word_end,
                            confidence=0.7  # Lower confidence for this fallback method
                        ))
        else:
            # Fallback for older Whisper versions or if no segments
            logger.warning("Whisper did not produce segment information. Using simple time distribution.")
            
            # Get the audio duration
            import soundfile as sf
            info = sf.info(audio_file)
            duration = info.duration
            
            # Split the text into words and distribute evenly
            words = text.split()
            word_duration = duration / len(words) if words else 0
            
            for i, word in enumerate(words):
                word_start = i * word_duration
                word_end = (i + 1) * word_duration
                
                word_timings.append(WordTiming(
                    word=word,
                    start_time=word_start,
                    end_time=word_end,
                    confidence=0.5  # Low confidence for this fallback method
                ))
        
        return word_timings
    
    def _gentle_align(self, audio_file: str, text: str) -> List[WordTiming]:
        """Align audio and text using Gentle."""
        gentle = self._get_aligner()
        requests = gentle["requests"]
        url = gentle["url"]
        
        # Prepare the request
        with open(audio_file, "rb") as audio_data:
            files = {
                "audio": audio_data,
                "transcript": (None, text)
            }
            
            # Send the request to the Gentle server
            response = requests.post(url, files=files)
            
            if response.status_code != 200:
                logger.error(f"Gentle server returned error: {response.status_code} {response.text}")
                return self._fallback_align(audio_file, text)
            
            # Process the response
            alignment = response.json()
            
            word_timings = []
            for word_data in alignment.get("words", []):
                if "start" in word_data and "end" in word_data:
                    word_timings.append(WordTiming(
                        word=word_data["word"],
                        start_time=word_data["start"],
                        end_time=word_data["end"],
                        confidence=word_data.get("confidence", 0.8)
                    ))
            
            return word_timings
    
    def _fallback_align(self, audio_file: str, text: str) -> List[WordTiming]:
        """Fallback method for alignment when other methods fail."""
        logger.warning("Using fallback alignment method (even distribution).")
        
        # Get the audio duration
        info = sf.info(audio_file)
        duration = info.duration
        
        # Split the text into words and distribute evenly
        words = text.split()
        word_duration = duration / len(words) if words else 0
        
        word_timings = []
        for i, word in enumerate(words):
            word_start = i * word_duration
            word_end = (i + 1) * word_duration
            
            word_timings.append(WordTiming(
                word=word,
                start_time=word_start,
                end_time=word_end,
                confidence=0.3  # Low confidence for this fallback method
            ))
        
        return word_timings
    
    def _get_cache_key(self, text: str, voice_id: str) -> str:
        """Generate a cache key for the given text and voice."""
        import hashlib
        
        # Create a hash from the text and voice
        text_hash = hashlib.md5(text.encode()).hexdigest()
        voice_hash = hashlib.md5(str(voice_id).encode()).hexdigest()
        
        return f"{self.config.engine.value}_{text_hash}_{voice_hash}"
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if the audio is already in cache."""
        if not self.config.use_cache:
            return None
            
        cache_dir = Path(self.config.cache_dir)
        audio_path = cache_dir / f"{cache_key}.wav"
        timing_path = cache_dir / f"{cache_key}.json"
        
        if audio_path.exists() and timing_path.exists():
            # Load the timing data
            with open(timing_path, "r") as f:
                timing_data = json.load(f)
                
            # Create result object
            result = NarrationResult(
                audio_path=str(audio_path),
                sample_rate=timing_data.get("sample_rate", 24000),
                duration=timing_data.get("duration", 0.0),
                word_timings=[
                    WordTiming(**word_data)
                    for word_data in timing_data.get("word_timings", [])
                ],
                text=timing_data.get("text", ""),
                voice_id=timing_data.get("voice_id", ""),
                tts_engine=TTSEngine(timing_data.get("tts_engine", self.config.engine.value)),
                metadata=timing_data.get("metadata", {})
            )
            
            logger.info(f"Found cached audio for '{text[:30]}...'")
            return result
            
        return None
    
    def _save_to_cache(self, cache_key: str, result: NarrationResult):
        """Save the generated audio and timing to cache."""
        if not self.config.use_cache:
            return
            
        cache_dir = Path(self.config.cache_dir)
        audio_path = cache_dir / f"{cache_key}.wav"
        timing_path = cache_dir / f"{cache_key}.json"
        
        # Copy the audio file if it's not already in the cache
        if str(audio_path) != result.audio_path:
            # Read from the original file and write to the cache
            audio_data, _ = sf.read(result.audio_path)
            sf.write(str(audio_path), audio_data, result.sample_rate)
        
        # Save the timing data
        timing_data = {
            "sample_rate": result.sample_rate,
            "duration": result.duration,
            "word_timings": [asdict(word) for word in result.word_timings],
            "text": result.text,
            "voice_id": result.voice_id,
            "tts_engine": result.tts_engine.value,
            "metadata": result.metadata
        }
        
        with open(timing_path, "w") as f:
            json.dump(timing_data, f, indent=2)
    
    def generate(self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None) -> NarrationResult:
        """
        Generate narration audio for the given text.
        
        Args:
            text: The text to convert to speech
            voice_id: Voice ID to use (depends on TTS engine)
            output_path: Path to save the generated audio file (optional)
            
        Returns:
            NarrationResult object with audio path and word timings
        """
        # Use default voice if not specified
        if voice_id is None:
            voice_id = self.config.voice_id
        
        # Detect emotions in the text
        emotions = self._detect_emotions(text)
        
        # Check cache first
        cache_key = self._get_cache_key(text, voice_id)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Calculate readability metrics for metadata
        readability_metrics = {
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "sentence_count": textstat.sentence_count(text),
            "word_count": len(text.split())
        }
        
        # Generate a unique output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_filename = f"narration_{timestamp}.wav"
            output_path = os.path.join(self.config.output_dir, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate audio based on the selected TTS engine
        start_time = time.time()
        if self.config.engine == TTSEngine.BARK:
            audio_array, sample_rate = self._generate_bark_audio(text, voice_id, output_path)
        elif self.config.engine == TTSEngine.COQUI:
            audio_array, sample_rate = self._generate_coqui_audio(text, voice_id, output_path)
        else:
            raise ValueError(f"Unsupported TTS engine: {self.config.engine}")
        
        generation_time = time.time() - start_time
        
        # Calculate audio duration
        audio_duration = len(audio_array) / sample_rate
        
        # Align the audio with the text
        start_time = time.time()
        if self.config.aligner == AlignerType.AENEAS:
            word_timings = self._aeneas_align(output_path, text)
        elif self.config.aligner == AlignerType.WHISPER:
            word_timings = self._whisper_align(self._get_aligner()["model"], output_path, text)
        elif self.config.aligner == AlignerType.GENTLE:
            word_timings = self._gentle_align(output_path, text)
        else:
            raise ValueError(f"Unsupported aligner: {self.config.aligner}")
        
        alignment_time = time.time() - start_time
        
        # Create result
        result = NarrationResult(
            audio_path=output_path,
            sample_rate=sample_rate,
            duration=audio_duration,
            word_timings=word_timings,
            text=text,
            voice_id=voice_id,
            tts_engine=self.config.engine,
            metadata={
                "emotions": emotions,
                "readability": readability_metrics,
                "generation_time": generation_time,
                "alignment_time": alignment_time,
                "total_time": generation_time + alignment_time
            }
        )
        
        # Save to cache
        self._save_to_cache(cache_key, result)
        
        logger.info(f"Generated {audio_duration:.2f} seconds of audio in {generation_time:.2f} seconds")
        logger.info(f"Aligned {len(word_timings)} words in {alignment_time:.2f} seconds")
        
        return result
    
    def save_timing_file(self, result: NarrationResult, output_path: Optional[str] = None) -> str:
        """
        Save word timing information to a JSON file.
        
        Args:
            result: NarrationResult from generate()
            output_path: Path to save the JSON file (optional)
            
        Returns:
            Path to the saved JSON file
        """
        if output_path is None:
            # Use the same name as the audio file but with .json extension
            audio_path = Path(result.audio_path)
            output_path = str(audio_path.with_suffix(".json"))
        
        # Create the timing data
        timing_data = {
            "text": result.text,
            "audio_file": os.path.basename(result.audio_path),
            "duration": result.duration,
            "sample_rate": result.sample_rate,
            "word_count": len(result.word_timings),
            "words": [
                {
                    "word": word.word,
                    "start": word.start_time,
                    "end": word.end_time,
                    "confidence": word.confidence
                }
                for word in result.word_timings
            ],
            "metadata": result.metadata
        }
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(timing_data, f, indent=2)
        
        logger.info(f"Saved timing information to {output_path}")
        return output_path

def main():
    """Command-line entry point for narration generator."""
    parser = argparse.ArgumentParser(description="Generate narration audio from text")
    
    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--text", type=str, help="Text to convert to speech")
    input_group.add_argument("--file", type=str, help="Text file to convert to speech")
    
    # TTS engine options
    tts_group = parser.add_argument_group("TTS Engine")
    tts_group.add_argument("--engine", type=str, choices=["bark", "coqui"], default="bark",
                          help="TTS engine to use")
    tts_group.add_argument("--voice", type=str, default="v2/en_speaker_6",
                          help="Voice ID to use (depends on TTS engine)")
    
    # Aligner options
    aligner_group = parser.add_argument_group("Aligner")
    aligner_group.add_argument("--aligner", type=str, choices=["aeneas", "whisper", "gentle"], default="aeneas",
                              help="Speech aligner to use")
    aligner_group.add_argument("--gentle-url", type=str, default="http://localhost:8765/transcriptions",
                              help="URL for Gentle server (only used with --aligner=gentle)")
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", type=str, help="Output directory")
    output_group.add_argument("--filename", type=str, help="Output filename")
    
    # Advanced options
    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument("--no-cache", action="store_true", help="Disable caching")
    advanced_group.add_argument("--cache-dir", type=str, default=".cache/narration", help="Cache directory")
    advanced_group.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    advanced_group.add_argument("--temperature", type=float, default=0.7, 
                               help="Temperature for generation (higher = more creative)")
    
    args = parser.parse_args()
    
    # Check input
    if args.text is None and args.file is None:
        print("Please provide text using --text or --file")
        parser.print_help()
        sys.exit(1)
    
    # Load text from file if specified
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        text = args.text
    
    # Create configuration
    config = NarrationConfig(
        engine=TTSEngine(args.engine),
        voice_id=args.voice,
        output_dir=args.output or "output/narration",
        aligner=AlignerType(args.aligner),
        gentle_url=args.gentle_url,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        emotion_enabled=not args.no_emotion,
        temperature=args.temperature
    )
    
    # Create narration generator
    generator = NarrationGenerator(
        tts_engine=config.engine,
        aligner=config.aligner,
        config=config,
        emotion_detection=config.emotion_enabled
    )
    
    # Generate narration
    output_path = None
    if args.filename:
        output_path = os.path.join(config.output_dir, args.filename)
    
    result = generator.generate(text, config.voice_id, output_path)
    
    # Save timing file
    timing_path = generator.save_timing_file(result)
    
    print(f"\nNarration generated successfully:")
    print(f"  Audio file: {result.audio_path}")
    print(f"  Timing file: {timing_path}")
    print(f"  Duration: {result.duration:.2f} seconds")
    print(f"  Word count: {len(result.word_timings)}")
    print(f"  Sample rate: {result.sample_rate} Hz")

if __name__ == "__main__":
    main()