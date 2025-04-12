#!/usr/bin/env python
"""
Setup script for the AI Video Teacher system.

This script handles the installation of all necessary components, including:
1. Core dependencies
2. TTS engines (Bark, Coqui)
3. Speech aligners (Aeneas, Whisper)

It also checks for required system dependencies (FFmpeg, eSpeak)
and provides instructions for installation if they are not found.
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
import setuptools
from setuptools import setup, find_packages

# Define the dependencies by component
DEPENDENCIES = {
    "core": [
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pytest>=6.2.5",
        "pydantic>=1.8.2",
        "langchain>=0.0.267",
        "sentence-transformers>=2.2.2",
        "spacy>=3.5.0",
        "python-docx>=0.8.11",
        "PyPDF2>=2.0.0",
        "scipy>=1.7.0"
    ],
    "script_parser": [
        "spacy>=3.5.0",
        "nltk>=3.6.0"
    ],
    "tts_engines": {
        "bark": [
            "bark @ git+https://github.com/suno-ai/bark.git",
            "torch>=1.13.0",
            "transformers>=4.25.1"
        ],
        "coqui": [
            "TTS>=0.13.0",
            "torch>=1.13.0"
        ]
    },
    "aligners": {
        "aeneas": ["aeneas>=1.7.3"],
        "gentle": [],  # Gentle is installed separately
        "whisper": [
            "openai-whisper>=20230124",
            "torch>=1.13.0",
            "ffmpeg-python>=0.2.0"
        ]
    }
}

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {platform.python_version()} ✓")


def check_command_exists(command):
    """Check if a command exists in PATH."""
    try:
        subprocess.run([command, "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=False)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def check_dependencies():
    """Check if required system dependencies are installed."""
    system = platform.system()
    missing_deps = []
    
    # Check FFmpeg
    if not check_command_exists("ffmpeg"):
        missing_deps.append("FFmpeg")
    else:
        print("FFmpeg found ✓")
    
    # Check eSpeak
    espeak_cmd = "espeak" if system != "Windows" else "espeak-ng"
    if not check_command_exists(espeak_cmd):
        missing_deps.append("eSpeak")
    else:
        print("eSpeak found ✓")
    
    if missing_deps:
        print("\nWARNING: The following dependencies are missing:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        print("\nInstallation instructions:")
        
        if "FFmpeg" in missing_deps:
            print("\nFFmpeg installation:")
            if system == "Windows":
                print("  1. Download FFmpeg from https://ffmpeg.org/download.html#build-windows")
                print("  2. Extract the archive and add the bin folder to your PATH")
            elif system == "Linux":
                print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
                print("  CentOS/RHEL: sudo yum install ffmpeg")
            elif system == "Darwin":  # macOS
                print("  With Homebrew: brew install ffmpeg")
        
        if "eSpeak" in missing_deps:
            print("\neSpeak installation:")
            if system == "Windows":
                print("  1. Download eSpeak-NG from https://github.com/espeak-ng/espeak-ng/releases")
                print("  2. Install and add to your PATH")
            elif system == "Linux":
                print("  Ubuntu/Debian: sudo apt-get install espeak")
                print("  CentOS/RHEL: sudo yum install espeak")
            elif system == "Darwin":  # macOS
                print("  With Homebrew: brew install espeak")
        
        return False
    
    return True


def install_spacy_model():
    """Install spaCy models."""
    try:
        print("\nInstalling spaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                      check=True)
        print("spaCy model installed successfully ✓")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error installing spaCy model: {e}")
        print("Please run manually: python -m spacy download en_core_web_sm")
        return False


def install_aeneas():
    """Install aeneas with error handling."""
    system = platform.system()
    
    print("\nInstalling aeneas...")
    print("This may take a while and requires system dependencies (FFmpeg and eSpeak).")
    
    try:
        # Install aeneas and its dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "aeneas>=1.7.3"], 
                      check=True)
        
        # Check if installation was successful by importing aeneas
        subprocess.run([sys.executable, "-c", "import aeneas; print('aeneas installation successful ✓')"], 
                      check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"\nError installing aeneas: {e}")
        print("\nManual installation instructions for aeneas:")
        
        if system == "Windows":
            print("1. Install FFmpeg and add to PATH")
            print("2. Install eSpeak-NG and add to PATH")
            print("3. Install aeneas from source: https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md#windows")
        elif system == "Linux":
            print("1. Install dependencies: sudo apt-get install ffmpeg espeak libespeak-dev")
            print("2. Install aeneas: pip install aeneas")
        elif system == "Darwin":  # macOS
            print("1. Install dependencies: brew install ffmpeg espeak")
            print("2. Install aeneas: pip install aeneas")
        
        print("\nAlternatively, you can use the Whisper aligner:")
        print("pip install openai-whisper")
        
        return False


def install_whisper():
    """Install OpenAI Whisper with error handling."""
    print("\nInstalling OpenAI Whisper...")
    
    try:
        # Install whisper and its dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], 
                      check=True)
        
        # Check if installation was successful
        subprocess.run([sys.executable, "-c", "import whisper; print('Whisper installation successful ✓')"], 
                      check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"\nError installing Whisper: {e}")
        print("Please install manually: pip install openai-whisper")
        return False


def install_dependencies():
    """Install all dependencies."""
    print("\nInstalling core dependencies...")
    core_deps = DEPENDENCIES["core"] + DEPENDENCIES["script_parser"]
    
    # Join dependencies into a single string for pip
    deps_str = " ".join(core_deps)
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *core_deps], 
                      check=True)
        print("Core dependencies installed successfully ✓")
    except subprocess.SubprocessError as e:
        print(f"Error installing core dependencies: {e}")
        return False
    
    # Ask user about TTS engines
    print("\nSelect TTS engines to install:")
    print("1. Bark (high quality but slow)")
    print("2. Coqui TTS (faster but less realistic)")
    print("3. Both")
    print("4. Skip TTS installation")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice in ("1", "3"):
        print("\nInstalling Bark TTS...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *DEPENDENCIES["tts_engines"]["bark"]], 
                          check=True)
            print("Bark TTS installed successfully ✓")
        except subprocess.SubprocessError as e:
            print(f"Error installing Bark TTS: {e}")
    
    if choice in ("2", "3"):
        print("\nInstalling Coqui TTS...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *DEPENDENCIES["tts_engines"]["coqui"]], 
                          check=True)
            print("Coqui TTS installed successfully ✓")
        except subprocess.SubprocessError as e:
            print(f"Error installing Coqui TTS: {e}")
    
    # Ask user about aligners
    print("\nSelect speech aligners to install:")
    print("1. aeneas (recommended, but has complex dependencies)")
    print("2. Whisper (easier to install, but less accurate)")
    print("3. Both")
    print("4. Skip aligner installation")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice in ("1", "3"):
        install_aeneas()
    
    if choice in ("2", "3"):
        install_whisper()
    
    # Install spaCy model
    install_spacy_model()
    
    return True


def create_requirements_file():
    """Create a requirements.txt file with all dependencies."""
    print("\nCreating requirements.txt file...")
    
    # Flatten dependencies
    requirements = []
    requirements.extend(DEPENDENCIES["core"])
    requirements.extend(DEPENDENCIES["script_parser"])
    
    # Add TTS engines
    for engine, deps in DEPENDENCIES["tts_engines"].items():
        requirements.extend(deps)
    
    # Add aligners
    for aligner, deps in DEPENDENCIES["aligners"].items():
        requirements.extend(deps)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_requirements = []
    for req in requirements:
        req_base = req.split(">=")[0].split("==")[0].strip()
        if req_base not in seen:
            seen.add(req_base)
            unique_requirements.append(req)
    
    # Write to file
    with open("requirements.txt", "w") as f:
        f.write("# AI Video Teacher requirements\n")
        f.write("# Generated by setup.py\n\n")
        for req in unique_requirements:
            f.write(f"{req}\n")
    
    print("requirements.txt created successfully ✓")


def main():
    """Main setup function."""
    print("=" * 70)
    print("AI Video Teacher System Setup")
    print("=" * 70)
    
    # Check Python version
    check_python_version()
    
    # Check system dependencies
    system_deps_ok = check_dependencies()
    if not system_deps_ok:
        print("\nWARNING: Some system dependencies are missing.")
        print("You can proceed with installation, but some features may not work.")
        proceed = input("Do you want to continue? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Setup aborted.")
            sys.exit(1)
    
    # Install dependencies
    install_ok = install_dependencies()
    
    # Create requirements.txt
    create_requirements_file()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Setup Summary")
    print("=" * 70)
    
    if install_ok:
        print("✓ Installation completed successfully")
    else:
        print("⚠ Installation completed with some issues")
    
    print("\nRecommended next steps:")
    print("1. Try running the demo script:")
    print("   python examples/demo_narration.py")
    print("2. Check the README files for more information on each component")
    print("3. Review the logs for any warnings or errors")
    
    print("\nThank you for installing AI Video Teacher!")
    print("=" * 70)


if __name__ == "__main__":
    main() 