#!/usr/bin/env python
"""
AI Video Teacher - Narration Generator (Production Version)

This script generates narrations for an educational script by:
1. Parsing the script into scenes
2. Generating audio narrations for each scene using a selected TTS engine
3. Creating alignment data for video generation using speech-to-text alignment
4. Outputting audio files and alignment JSON for each scene

Features:
- Multiple TTS engines support (Bark, Coqui)
- Multiple aligners support (Aeneas, Whisper)
- Comprehensive error handling and recovery
- Progress reporting with ETA
- Logging and detailed output summary
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("narration_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("narration_generator")

# Import dependencies
try:
    # Import script parser
    from src.script_parser.core import ScriptParser
    
    # Other utilities
    import torch
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please install required dependencies using setup.py")
    sys.exit(1)

# Create argument parser
def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="Generate narrations for an educational script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Group Input/Output arguments
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--script", "-s", 
        required=True,
        help="Path to the script file (.md, .docx, .pdf, .txt)"
    )
    io_group.add_argument(
        "--output", "-o",
        default="output",
        help="Directory to save generated narrations and alignment data"
    )
    io_group.add_argument(
        "--format", 
        choices=["wav", "mp3", "ogg"],
        default="wav",
        help="Audio format for output files"
    )
    io_group.add_argument(
        "--sample-rate", 
        type=int,
        default=24000,
        help="Sample rate for audio output"
    )
    
    # Group TTS Engine arguments
    tts_group = parser.add_argument_group("TTS Engine")
    tts_group.add_argument(
        "--tts-engine", 
        choices=["bark", "coqui"],
        default="bark",
        help="Text-to-speech engine to use"
    )
    tts_group.add_argument(
        "--voice", 
        default="v2/en_speaker_6",
        help="Voice ID for the selected TTS engine"
    )
    tts_group.add_argument(
        "--emotion",
        default="neutral",
        choices=["neutral", "happy", "sad", "excited", "authoritative"],
        help="Emotional tone for narration (if supported by TTS engine)"
    )
    tts_group.add_argument(
        "--no-variations",
        action="store_true",
        help="Disable emotional variations for narration"
    )
    
    # Group Speech Aligner arguments
    aligner_group = parser.add_argument_group("Speech Aligner")
    aligner_group.add_argument(
        "--aligner", 
        choices=["aeneas", "whisper", "none"],
        default="aeneas",
        help="Tool for aligning speech with text"
    )
    aligner_group.add_argument(
        "--alignment-granularity",
        choices=["word", "sentence", "paragraph"],
        default="word",
        help="Level of detail for alignment"
    )
    
    # Group Processing arguments
    processing_group = parser.add_argument_group("Processing")
    processing_group.add_argument(
        "--max-workers", 
        type=int,
        default=1,
        help="Maximum number of workers for parallel processing"
    )
    processing_group.add_argument(
        "--batch-size", 
        type=int,
        default=1,
        help="Batch size for TTS processing"
    )
    processing_group.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for TTS processing"
    )
    processing_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation for scenes that already have audio files"
    )
    
    # Group Debug arguments
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    debug_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse script but do not generate audio"
    )
    
    return parser

def check_dependencies(args: argparse.Namespace) -> bool:
    """
    Check if all required dependencies are installed based on selected options.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    # Check TTS engine dependencies
    if args.tts_engine == "bark":
        try:
            import bark
            logger.info("Bark TTS engine is available")
        except ImportError:
            logger.error("Bark TTS engine is not installed")
            logger.error("Install it with: pip install bark @ git+https://github.com/suno-ai/bark.git")
            return False
    
    elif args.tts_engine == "coqui":
        try:
            import TTS
            logger.info("Coqui TTS engine is available")
        except ImportError:
            logger.error("Coqui TTS engine is not installed")
            logger.error("Install it with: pip install TTS")
            return False
    
    # Check aligner dependencies
    if args.aligner == "aeneas":
        try:
            import aeneas
            logger.info("Aeneas aligner is available")
        except ImportError:
            logger.error("Aeneas aligner is not installed")
            logger.error("Install it with: pip install aeneas")
            logger.error("or use --aligner whisper instead")
            return False
    
    elif args.aligner == "whisper":
        try:
            import whisper
            logger.info("Whisper aligner is available")
        except ImportError:
            logger.error("Whisper aligner is not installed")
            logger.error("Install it with: pip install openai-whisper")
            return False
    
    return True

def setup_tts_engine(args: argparse.Namespace) -> Any:
    """
    Set up the selected TTS engine.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Any: Initialized TTS engine
    """
    if args.tts_engine == "bark":
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            from bark.generation import generate_text_semantic
            from bark.api import semantic_to_waveform
            
            logger.info("Setting up Bark TTS engine...")
            logger.info(f"Using device: {args.device}")
            
            # Preload models
            preload_models()
            
            # Create a simple wrapper for consistent API
            class BarkTTS:
                def __init__(self, voice_id: str, device: str):
                    self.voice_id = voice_id
                    self.device = device
                    self.sample_rate = SAMPLE_RATE
                
                def generate(self, text: str, emotion: str = "neutral") -> np.ndarray:
                    """Generate audio from text using Bark."""
                    # Add emotion prompt if not neutral
                    if emotion != "neutral" and not args.no_variations:
                        prompt_map = {
                            "happy": "[cheerful voice] ",
                            "sad": "[somber voice] ",
                            "excited": "[energetic voice] ",
                            "authoritative": "[authoritative voice] "
                        }
                        text = prompt_map.get(emotion, "") + text
                    
                    return generate_audio(text, history_prompt=self.voice_id)
            
            return BarkTTS(args.voice, args.device)
            
        except Exception as e:
            logger.error(f"Error setting up Bark TTS engine: {e}")
            raise
    
    elif args.tts_engine == "coqui":
        try:
            from TTS.api import TTS
            
            logger.info("Setting up Coqui TTS engine...")
            logger.info(f"Using device: {args.device}")
            
            # List available models
            if args.verbose:
                models = TTS().list_models()
                logger.info(f"Available Coqui TTS models: {models}")
            
            # Select appropriate model
            model = "tts_models/en/vctk/vits"  # Default model
            
            # Create TTS instance
            tts = TTS(model, gpu=(args.device == "cuda"))
            
            # Create a wrapper for consistent API
            class CoquiTTSWrapper:
                def __init__(self, tts_engine: TTS, voice_id: str):
                    self.tts = tts_engine
                    self.voice_id = voice_id
                    self.sample_rate = args.sample_rate
                
                def generate(self, text: str, emotion: str = "neutral") -> np.ndarray:
                    """Generate audio from text using Coqui TTS."""
                    # For multi-speaker models, select speaker
                    speaker = self.voice_id if self.tts.is_multi_speaker else None
                    
                    # Generate and return audio
                    wav = self.tts.tts(text, speaker=speaker)
                    return np.array(wav)
            
            return CoquiTTSWrapper(tts, args.voice)
            
        except Exception as e:
            logger.error(f"Error setting up Coqui TTS engine: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported TTS engine: {args.tts_engine}")

def setup_speech_aligner(args: argparse.Namespace) -> Any:
    """
    Set up the selected speech aligner.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Any: Initialized speech aligner
    """
    if args.aligner == "aeneas":
        try:
            from aeneas.executetask import ExecuteTask
            from aeneas.task import Task
            
            logger.info("Setting up Aeneas speech aligner...")
            
            # Create a wrapper for consistent API
            class AeneasAligner:
                def __init__(self, granularity: str):
                    self.granularity = granularity
                
                def align(self, text: str, audio_path: str) -> List[Dict]:
                    """Align text to audio using Aeneas."""
                    # Create a temporary file with the text
                    temp_txt = f"{audio_path}.txt"
                    with open(temp_txt, "w", encoding="utf-8") as f:
                        f.write(text)
                    
                    # Set up Aeneas task
                    config = f"task_language=eng|os_task_file_format=json|is_text_type=plain|is_audio_file_detect_head_max=0|task_adjust_boundary_algorithm=auto|task_adjust_boundary_nonspeech_min=0.500|task_adjust_boundary_nonspeech_string=REMOVE|is_audio_file_detect_head_min=0.000"
                    
                    # Add granularity
                    if self.granularity == "word":
                        config += "|is_text_unparsed_id_regex=\\w+"
                        # Prepare word-level text
                        with open(temp_txt, "w", encoding="utf-8") as f:
                            words = text.split()
                            for i, word in enumerate(words):
                                f.write(f"{i+1} {word}\n")
                    elif self.granularity == "sentence":
                        config += "|is_text_unparsed_id_regex=\\d+"
                        # Prepare sentence-level text
                        with open(temp_txt, "w", encoding="utf-8") as f:
                            sentences = text.split(". ")
                            for i, sentence in enumerate(sentences):
                                f.write(f"{i+1} {sentence}.\n")
                    
                    # Create task
                    task = Task(config_string=config)
                    task.audio_file_path_absolute = audio_path
                    task.text_file_path_absolute = temp_txt
                    task.sync_map_file_path_absolute = f"{audio_path}.json"
                    
                    # Execute task
                    executor = ExecuteTask(task)
                    result = executor.execute()
                    
                    if result:
                        # Parse the output JSON
                        with open(f"{audio_path}.json", "r", encoding="utf-8") as f:
                            alignment_data = json.load(f)
                        
                        # Format the alignment data
                        fragments = []
                        for fragment in alignment_data.get("fragments", []):
                            fragments.append({
                                "begin": float(fragment.get("begin", 0)),
                                "end": float(fragment.get("end", 0)),
                                "text": fragment.get("lines", [""])[0]
                            })
                        
                        # Clean up temp file
                        os.remove(temp_txt)
                        os.remove(f"{audio_path}.json")
                        
                        return fragments
                    else:
                        logger.error(f"Alignment failed for {audio_path}")
                        return []
            
            return AeneasAligner(args.alignment_granularity)
            
        except Exception as e:
            logger.error(f"Error setting up Aeneas speech aligner: {e}")
            raise
    
    elif args.aligner == "whisper":
        try:
            import whisper
            
            logger.info("Setting up Whisper speech aligner...")
            logger.info(f"Using device: {args.device}")
            
            # Load whisper model - use small for better speed/accuracy tradeoff
            model = whisper.load_model("small", device=args.device)
            
            # Create a wrapper for consistent API
            class WhisperAligner:
                def __init__(self, model: Any, granularity: str):
                    self.model = model
                    self.granularity = granularity
                
                def align(self, text: str, audio_path: str) -> List[Dict]:
                    """Align text to audio using Whisper."""
                    # Transcribe audio with word timestamps
                    result = self.model.transcribe(
                        audio_path, 
                        word_timestamps=True,
                        language="en"
                    )
                    
                    # Format the alignment data based on granularity
                    if self.granularity == "word":
                        # Word-level alignment
                        fragments = []
                        for segment in result["segments"]:
                            for word in segment.get("words", []):
                                fragments.append({
                                    "begin": word["start"],
                                    "end": word["end"],
                                    "text": word["word"].strip()
                                })
                    
                    elif self.granularity == "sentence":
                        # Sentence-level alignment
                        fragments = []
                        for segment in result["segments"]:
                            fragments.append({
                                "begin": segment["start"],
                                "end": segment["end"],
                                "text": segment["text"].strip()
                            })
                    
                    else:  # paragraph
                        # Paragraph-level alignment (groups segments)
                        fragments = []
                        if result["segments"]:
                            current_paragraph = {
                                "begin": result["segments"][0]["start"],
                                "end": result["segments"][0]["end"],
                                "text": result["segments"][0]["text"].strip()
                            }
                            
                            for segment in result["segments"][1:]:
                                # Assume new paragraph if pause > 1 second
                                if segment["start"] - current_paragraph["end"] > 1.0:
                                    fragments.append(current_paragraph)
                                    current_paragraph = {
                                        "begin": segment["start"],
                                        "end": segment["end"],
                                        "text": segment["text"].strip()
                                    }
                                else:
                                    current_paragraph["end"] = segment["end"]
                                    current_paragraph["text"] += " " + segment["text"].strip()
                            
                            fragments.append(current_paragraph)
                    
                    return fragments
            
            return WhisperAligner(model, args.alignment_granularity)
            
        except Exception as e:
            logger.error(f"Error setting up Whisper speech aligner: {e}")
            raise
    
    elif args.aligner == "none":
        # Create a dummy aligner
        class DummyAligner:
            def __init__(self):
                pass
            
            def align(self, text: str, audio_path: str) -> List[Dict]:
                """Return empty alignment data."""
                return []
        
        return DummyAligner()
    
    else:
        raise ValueError(f"Unsupported speech aligner: {args.aligner}")

def process_script(args: argparse.Namespace) -> Tuple[List, Dict]:
    """
    Process the script file and extract scenes.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple[List, Dict]: List of parsed scenes and metadata
    """
    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")
    
    logger.info(f"Processing script: {script_path}")
    
    # Initialize script parser
    parser = ScriptParser()
    
    # Process script based on file extension
    extension = script_path.suffix.lower()
    
    scenes = []
    metadata = {}
    
    try:
        if extension == ".md":
            logger.info("Parsing Markdown script...")
            scenes = parser.parse_markdown(str(script_path))
        elif extension == ".docx":
            logger.info("Parsing Word document...")
            scenes = parser.parse_docx(str(script_path))
        elif extension == ".pdf":
            logger.info("Parsing PDF document...")
            scenes = parser.parse_pdf(str(script_path))
        elif extension == ".txt" or True:  # Default to txt
            logger.info("Parsing text document...")
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
            scenes = parser.parse_text(content)
        
        # Extract metadata
        metadata = {
            "title": script_path.stem,
            "num_scenes": len(scenes),
            "date_processed": datetime.now().isoformat(),
            "script_file": str(script_path)
        }
        
        logger.info(f"Successfully parsed {len(scenes)} scenes from script")
        
        if args.verbose:
            for i, scene in enumerate(scenes):
                logger.info(f"Scene {i+1}: {scene.title} - {scene.scene_type.name} ({len(scene.content_blocks)} blocks)")
        
        return scenes, metadata
        
    except Exception as e:
        logger.error(f"Error parsing script: {e}")
        logger.error(traceback.format_exc())
        raise

def generate_narration(
    scene: Any, 
    tts_engine: Any,
    index: int,
    output_dir: Path,
    args: argparse.Namespace
) -> Tuple[str, Dict]:
    """
    Generate narration for a scene.
    
    Args:
        scene: Scene object with content
        tts_engine: Initialized TTS engine
        index: Scene index
        output_dir: Output directory
        args: Parsed command line arguments
        
    Returns:
        Tuple[str, Dict]: Path to output audio file and metadata
    """
    # Create scene-specific output directory
    scene_dir = output_dir / f"scene_{index:03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    audio_file = scene_dir / f"narration.{args.format}"
    
    # Skip if file exists and --skip-existing is set
    if audio_file.exists() and args.skip_existing:
        logger.info(f"Skipping scene {index} narration generation (file exists)")
        # Load metadata if available
        metadata_file = scene_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return str(audio_file), metadata
    
    # Extract text from scene
    text = ""
    for block in scene.content_blocks:
        if block.content_type.name in ["NARRATION", "EXPLANATION", "QUOTE", "DEFINITION"]:
            text += block.text + " "
    
    # Clean up text
    text = text.strip()
    if not text:
        logger.warning(f"No narration text found in scene {index}")
        text = f"Scene {index}: {scene.title if hasattr(scene, 'title') else 'Untitled'}"
    
    try:
        # Log generation details
        logger.info(f"Generating narration for scene {index} ({len(text)} chars)")
        
        # Select emotion based on scene type
        emotion = args.emotion
        if not args.no_variations:
            if hasattr(scene, 'scene_type'):
                type_emotion_map = {
                    "INTRO": "excited",
                    "RECAP": "neutral",
                    "THEORY": "authoritative",
                    "EXAMPLE": "happy",
                    "SUMMARY": "neutral"
                }
                emotion = type_emotion_map.get(scene.scene_type.name, args.emotion)
        
        # Generate audio
        start_time = time.time()
        audio = tts_engine.generate(text, emotion)
        generation_time = time.time() - start_time
        
        # Save audio
        import scipy.io.wavfile as wavfile
        wavfile.write(audio_file, tts_engine.sample_rate, audio)
        
        # Create metadata
        metadata = {
            "scene_index": index,
            "audio_file": str(audio_file),
            "text": text,
            "duration": len(audio) / tts_engine.sample_rate,
            "generation_time": generation_time,
            "tts_engine": args.tts_engine,
            "emotion": emotion
        }
        
        # Save metadata
        with open(scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return str(audio_file), metadata
    
    except Exception as e:
        logger.error(f"Error generating narration for scene {index}: {e}")
        logger.error(traceback.format_exc())
        raise

def generate_alignment(
    scene: Any,
    aligner: Any, 
    audio_file: str,
    text: str,
    scene_dir: Path,
    args: argparse.Namespace
) -> Dict:
    """
    Generate alignment data for a scene.
    
    Args:
        scene: Scene object with content
        aligner: Initialized speech aligner
        audio_file: Path to audio file
        text: Text that was narrated
        scene_dir: Output directory for the scene
        args: Parsed command line arguments
        
    Returns:
        Dict: Alignment data
    """
    if args.aligner == "none":
        logger.info(f"Skipping alignment generation (aligner=none)")
        return {}
    
    # Determine output alignment file
    alignment_file = Path(scene_dir) / "alignment.json"
    
    # Skip if file exists and --skip-existing is set
    if alignment_file.exists() and args.skip_existing:
        logger.info(f"Skipping alignment generation (file exists)")
        with open(alignment_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    try:
        # Generate alignment
        logger.info(f"Generating alignment for {os.path.basename(audio_file)}")
        start_time = time.time()
        
        fragments = aligner.align(text, audio_file)
        
        alignment_time = time.time() - start_time
        
        # Create alignment data
        alignment_data = {
            "audio_file": audio_file,
            "fragments": fragments,
            "alignment_time": alignment_time,
            "aligner": args.aligner,
            "granularity": args.alignment_granularity
        }
        
        # Save alignment data
        with open(alignment_file, "w", encoding="utf-8") as f:
            json.dump(alignment_data, f, indent=2)
        
        return alignment_data
    
    except Exception as e:
        logger.error(f"Error generating alignment: {e}")
        logger.error(traceback.format_exc())
        # Return empty alignment
        return {"fragments": [], "error": str(e)}

def main():
    """Main function to run the narration generator."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Log settings
    logger.info(f"Script: {args.script}")
    logger.info(f"Output: {args.output}")
    logger.info(f"TTS Engine: {args.tts_engine}")
    logger.info(f"Voice: {args.voice}")
    logger.info(f"Aligner: {args.aligner}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Check dependencies
        if not check_dependencies(args):
            logger.error("Missing dependencies. Please install them and try again.")
            return 1
        
        # Process script
        scenes, script_metadata = process_script(args)
        
        if args.dry_run:
            logger.info("Dry run completed. Exiting without generating audio.")
            return 0
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TTS engine
        tts_engine = setup_tts_engine(args)
        
        # Setup speech aligner
        aligner = setup_speech_aligner(args)
        
        # Generate narrations for each scene
        narrations = []
        with tqdm(total=len(scenes), desc="Generating narrations") as pbar:
            for i, scene in enumerate(scenes):
                try:
                    # Generate narration
                    audio_file, metadata = generate_narration(
                        scene, tts_engine, i, output_dir, args
                    )
                    
                    # Generate alignment
                    alignment_data = generate_alignment(
                        scene, aligner, audio_file, metadata["text"], 
                        output_dir / f"scene_{i:03d}", args
                    )
                    
                    # Combine metadata
                    scene_data = {
                        "index": i,
                        "title": scene.title if hasattr(scene, 'title') else f"Scene {i}",
                        "audio_file": audio_file,
                        "metadata": metadata,
                        "alignment": alignment_data
                    }
                    
                    narrations.append(scene_data)
                    
                except Exception as e:
                    logger.error(f"Error processing scene {i}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Add failed scene
                    narrations.append({
                        "index": i,
                        "title": scene.title if hasattr(scene, 'title') else f"Scene {i}",
                        "error": str(e)
                    })
                
                pbar.update(1)
        
        # Save summary
        summary = {
            "script": args.script,
            "output_dir": str(output_dir),
            "num_scenes": len(scenes),
            "num_narrations": sum(1 for n in narrations if "error" not in n),
            "num_failed": sum(1 for n in narrations if "error" in n),
            "tts_engine": args.tts_engine,
            "aligner": args.aligner,
            "voice": args.voice,
            "date": datetime.now().isoformat(),
            "narrations": narrations
        }
        
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        # Print completion message
        success_rate = summary["num_narrations"] / max(1, len(scenes)) * 100
        logger.info(f"Narration generation completed: {summary['num_narrations']}/{len(scenes)} scenes successful ({success_rate:.1f}%)")
        
        if summary["num_failed"] > 0:
            logger.warning(f"Failed to generate narrations for {summary['num_failed']} scenes")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 