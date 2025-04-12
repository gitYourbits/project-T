# AI Video Teacher - Narration Generator

The Narration Generator is a key component of the AI Video Teacher system responsible for converting text scripts into professional-quality narration audio with precise timing information.

## Features

- **Multiple TTS Engines**:
  - Bark TTS: High-quality, natural-sounding voices with emotion support
  - Coqui TTS: Open-source TTS system with multiple voices and languages

- **Audio-Text Alignment**:
  - Aeneas: Forced alignment for precise word timings
  - Whisper: Speech-to-text based alignment
  - Gentle: Alternative forced alignment with additional features

- **Emotion Detection and Application**:
  - Automatically detects emotional context in text
  - Applies appropriate voice modulation based on detected emotions

- **Output Formats**:
  - Audio files (WAV) for each scene
  - JSON timing files with word-level synchronization data
  - Complete narration metadata for video rendering

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)
- CUDA-capable GPU (optional, but recommended for faster processing)

### Dependency Installation

We provide a dedicated installation script that handles all dependencies:

```bash
# Basic installation (CPU only)
python src/narration_generator/install_dependencies.py

# Installation with CUDA support
python src/narration_generator/install_dependencies.py --cuda

# Install only specific components
python src/narration_generator/install_dependencies.py --tts bark --aligner whisper
```

For detailed installation options:

```bash
python src/narration_generator/install_dependencies.py --help
```

### Special Note About Aeneas

Aeneas requires additional system dependencies that can be challenging to install on some platforms:

- **Windows**: Follow the Windows-specific instructions provided by the installer
- **macOS**: Requires espeak and ffmpeg (can be installed via Homebrew)
- **Linux**: Requires espeak, libespeak-dev, and ffmpeg packages

## Usage

```python
from src.narration_generator.core import NarrationGenerator, NarrationConfig, TTSEngine

# Initialize the generator
narrator = NarrationGenerator(
    tts_engine=TTSEngine.BARK,  # or TTSEngine.COQUI
    aligner="aeneas",           # or "whisper" or "gentle"
    use_emotions=True,          # Enable emotion detection and application
    cache_dir="./tts_cache"     # Optional cache for faster processing
)

# Generate narration for a parsed script
narration_results = narrator.generate_narration_for_script(script_data)

# Access results
for scene_id, scene_narration in narration_results.items():
    print(f"Scene {scene_id}:")
    print(f"  Audio file: {scene_narration.audio_file}")
    print(f"  Duration: {scene_narration.duration:.2f} seconds")
    print(f"  Word timings: {len(scene_narration.word_timings)} words aligned")
```

## API Reference

### Core Classes

- `NarrationGenerator`: Main class for generating narrations
- `NarrationConfig`: Configuration options for the narration process
- `TTSEngine`: Enum of supported TTS engines
- `SpeechAligner`: Handles alignment of audio with text

### Key Methods

- `generate_narration_for_scene`: Process a single scene
- `generate_narration_for_script`: Process an entire script
- `get_tts_engine`: Get the currently configured TTS engine
- `parse_emotions`: Extract emotional context from text

## Troubleshooting

If you encounter issues with any of the TTS engines or aligners, please check:

1. All dependencies are correctly installed
2. Your system meets the requirements (especially for GPU acceleration)
3. You have sufficient disk space for the audio files and models

For Aeneas-specific issues, refer to the [official Aeneas documentation](https://github.com/readbeyond/aeneas/blob/master/wiki/TROUBLESHOOTING.md).

## License

This component is part of the AI Video Teacher system and is subject to the same license terms as the overall project. 