"""
Speech-to-text alignment interfaces and implementations.

This module provides tools for aligning generated speech with the source text,
enabling precise synchronization between narration and visual elements.
"""

import os
import abc
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class SpeechAligner(abc.ABC):
    """Abstract base class for speech-to-text aligners."""
    
    @abc.abstractmethod
    def align(self, audio_path: str, text: str) -> List[Dict[str, Any]]:
        """
        Align speech audio with text.
        
        Args:
            audio_path: Path to the audio file to align
            text: Text to align with the audio
            
        Returns:
            List of dictionaries containing word/phrase alignments
        """
        pass
    
    def save_alignment(self, alignment: List[Dict[str, Any]], output_path: str) -> str:
        """
        Save alignment data to a JSON file.
        
        Args:
            alignment: Alignment data
            output_path: Path to save the alignment data
            
        Returns:
            Path to the saved alignment file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save alignment data
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alignment, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved alignment data to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save alignment data to {output_path}: {e}")
            raise RuntimeError(f"Failed to save alignment data: {e}") from e
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the speech aligner."""
        pass


class AeneasAligner(SpeechAligner):
    """
    Aeneas-based speech-to-text aligner.
    
    Aeneas is a Python/C library for forced alignment of audio and text,
    developed for automatic synchronization of audio books.
    """
    
    def __init__(self):
        """Initialize the Aeneas aligner."""
        # Check if aeneas is installed
        try:
            import aeneas
            self.aeneas = aeneas
            logger.info("Aeneas aligner initialized")
        except ImportError as e:
            logger.error(f"Failed to import Aeneas: {e}")
            logger.error("Please install with: pip install aeneas")
            raise ImportError("Aeneas not installed") from e
    
    def align(self, audio_path: str, text: str) -> List[Dict[str, Any]]:
        """
        Align speech audio with text using Aeneas.
        
        Args:
            audio_path: Path to the audio file to align
            text: Text to align with the audio
            
        Returns:
            List of dictionaries containing word/phrase alignments
        """
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
        from aeneas.textfile import TextFile
        from aeneas.language import Language
        
        try:
            # Create a temporary file for the text
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', encoding='utf-8', delete=False) as f:
                text_path = f.name
                # Split text into sentences or phrases for better alignment
                lines = self._split_text_for_alignment(text)
                for i, line in enumerate(lines):
                    f.write(f"{i+1}|{line}\n")
            
            # Configure the task
            config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"
            task = Task(config_string=config_string)
            task.audio_file_path_absolute = audio_path
            task.text_file_path_absolute = text_path
            
            # Execute the task
            ExecuteTask(task).execute()
            
            # Process the alignment result
            alignment = []
            for fragment in task.sync_map.fragments:
                begin = fragment.begin
                end = fragment.end
                text = fragment.text.strip().split('|', 1)[1] if '|' in fragment.text else fragment.text.strip()
                
                alignment.append({
                    "start": round(begin, 3),
                    "end": round(end, 3),
                    "text": text
                })
            
            # Clean up the temporary file
            try:
                os.remove(text_path)
            except:
                pass
            
            return alignment
            
        except Exception as e:
            logger.error(f"Error aligning with Aeneas: {e}")
            raise RuntimeError(f"Failed to align speech with Aeneas: {e}") from e
    
    def _split_text_for_alignment(self, text: str) -> List[str]:
        """
        Split text into smaller chunks for better alignment.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split by sentences (simple approach)
        import re
        
        # First handle common abbreviations to avoid incorrect splits
        abbr_text = text
        common_abbrs = ["Mr.", "Mrs.", "Dr.", "e.g.", "i.e.", "etc.", "vs.", "U.S.A."]
        for abbr in common_abbrs:
            abbr_text = abbr_text.replace(abbr, abbr.replace(".", "<period>"))
        
        # Split by sentence endings
        chunks = re.split(r'(?<=[.!?])\s+', abbr_text)
        
        # Restore periods in abbreviations
        chunks = [chunk.replace("<period>", ".") for chunk in chunks]
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    @property
    def name(self) -> str:
        """Return the name of the speech aligner."""
        return "Aeneas"


class GentleAligner(SpeechAligner):
    """
    Gentle-based speech-to-text aligner.
    
    Gentle is a robust yet lenient forced aligner built on Kaldi,
    providing word-level alignment.
    """
    
    def __init__(self, gentle_path: Optional[str] = None):
        """
        Initialize the Gentle aligner.
        
        Args:
            gentle_path: Optional path to the Gentle installation directory
        """
        self.gentle_path = gentle_path
        
        # Check if Gentle is accessible
        if not self._is_gentle_installed():
            logger.warning("Gentle may not be installed or accessible.")
            logger.warning("Please install Gentle from https://github.com/lowerquality/gentle")
            logger.warning("To use Gentle, either add it to your PATH or specify gentle_path")
    
    def _is_gentle_installed(self) -> bool:
        """Check if Gentle is installed."""
        if self.gentle_path:
            # Check if the specified path exists
            return os.path.exists(os.path.join(self.gentle_path, "align.py"))
        else:
            # Check if gentle is in PATH
            try:
                subprocess.run(["which", "gentle"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                return True
            except:
                try:
                    # On Windows
                    subprocess.run(["where", "gentle"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    return True
                except:
                    return False
    
    def align(self, audio_path: str, text: str) -> List[Dict[str, Any]]:
        """
        Align speech audio with text using Gentle.
        
        Args:
            audio_path: Path to the audio file to align
            text: Text to align with the audio
            
        Returns:
            List of dictionaries containing word alignments
        """
        try:
            # Create a temporary file for the text
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', encoding='utf-8', delete=False) as f:
                text_path = f.name
                f.write(text)
            
            # Create a temporary file for the alignment output
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                output_path = f.name
            
            # Build the command
            if self.gentle_path:
                cmd = ["python", os.path.join(self.gentle_path, "align.py")]
            else:
                cmd = ["gentle"]
            
            cmd.extend([
                audio_path,
                text_path,
                "-o", output_path
            ])
            
            # Run Gentle
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"Gentle alignment failed: {process.stderr}")
                raise RuntimeError(f"Gentle alignment failed: {process.stderr}")
            
            # Read and process the alignment result
            with open(output_path, 'r', encoding='utf-8') as f:
                gentle_result = json.load(f)
            
            # Convert Gentle format to our format
            alignment = []
            
            for word in gentle_result.get("words", []):
                if "start" in word and "end" in word:
                    alignment.append({
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                        "text": word["word"],
                        "confidence": word.get("confidence", 1.0)
                    })
            
            # Clean up temporary files
            try:
                os.remove(text_path)
                os.remove(output_path)
            except:
                pass
            
            return alignment
            
        except Exception as e:
            logger.error(f"Error aligning with Gentle: {e}")
            raise RuntimeError(f"Failed to align speech with Gentle: {e}") from e
    
    @property
    def name(self) -> str:
        """Return the name of the speech aligner."""
        return "Gentle"


class WhisperAligner(SpeechAligner):
    """
    OpenAI Whisper-based speech-to-text aligner.
    
    Uses Whisper's capabilities for transcription and alignment.
    """
    
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        """
        Initialize the Whisper aligner.
        
        Args:
            model_name: Whisper model to use
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
                
            logger.info(f"Initializing Whisper aligner (device: {self.device}, model: {self.model_name})")
            
        except ImportError as e:
            logger.error(f"Failed to import Whisper dependencies: {e}")
            logger.error("Please install with: pip install openai-whisper")
            raise ImportError("Whisper dependencies not installed") from e
    
    def _ensure_loaded(self):
        """Ensure the model is loaded."""
        if not self.loaded:
            try:
                import whisper
                
                logger.info(f"Loading Whisper model: {self.model_name}...")
                self.model = whisper.load_model(self.model_name, device=self.device)
                self.loaded = True
                logger.info("Whisper model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise RuntimeError("Failed to load Whisper model") from e
    
    def align(self, audio_path: str, text: str) -> List[Dict[str, Any]]:
        """
        Align speech audio with text using Whisper.
        
        Args:
            audio_path: Path to the audio file to align
            text: Text to align with the audio
            
        Returns:
            List of dictionaries containing word/segment alignments
        """
        self._ensure_loaded()
        
        try:
            # Load audio with whisper
            import whisper
            import warnings
            
            # Suppress warnings during audio loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = whisper.load_audio(audio_path)
                
            # Run transcription with word timestamps
            result = self.model.transcribe(
                audio, 
                language="en", 
                word_timestamps=True,  # This may not be available in older whisper versions
                initial_prompt=text    # Use the provided text as a prompt for better alignment
            )
            
            # Extract word-level alignments when available
            alignment = []
            
            # Some whisper versions may not support word timestamps
            # In that case, fall back to segment timestamps
            if "words" in result and result["words"]:
                # Word-level alignment
                for word in result["words"]:
                    alignment.append({
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                        "text": word["word"],
                        "confidence": word.get("probability", 1.0)
                    })
            else:
                # Segment-level alignment
                for segment in result["segments"]:
                    alignment.append({
                        "start": round(segment["start"], 3),
                        "end": round(segment["end"], 3),
                        "text": segment["text"].strip(),
                        "confidence": segment.get("confidence", 1.0)
                    })
            
            return alignment
            
        except Exception as e:
            logger.error(f"Error aligning with Whisper: {e}")
            raise RuntimeError(f"Failed to align speech with Whisper: {e}") from e
    
    @property
    def name(self) -> str:
        """Return the name of the speech aligner."""
        return f"Whisper ({self.model_name})"


# Factory function to create speech aligners
def create_speech_aligner(aligner_type: str, **kwargs) -> SpeechAligner:
    """
    Factory function to create a speech aligner.
    
    Args:
        aligner_type: Type of speech aligner to create
        **kwargs: Additional arguments to pass to the aligner constructor
        
    Returns:
        Initialized speech aligner
    """
    aligners = {
        "aeneas": AeneasAligner,
        "gentle": GentleAligner,
        "whisper": WhisperAligner,
    }
    
    if aligner_type not in aligners:
        raise ValueError(f"Unknown speech aligner type: {aligner_type}. Available types: {list(aligners.keys())}")
    
    return aligners[aligner_type](**kwargs) 