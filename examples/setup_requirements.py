#!/usr/bin/env python
"""
AI Video Teacher - Narration Generator Requirements Setup

This script helps install all necessary dependencies for the narration generator.
It handles the complex dependencies for different TTS engines and speech aligners.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

# Define the requirements for each component
BASIC_REQUIREMENTS = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "tqdm>=4.62.0",
    "requests>=2.25.0",
    "pydub>=0.25.1",
    "librosa>=0.9.0",
    "soundfile>=0.10.3",
    "matplotlib>=3.4.0",
]

BARK_REQUIREMENTS = [
    "torch>=1.10.0",
    "transformers>=4.25.0",
    "accelerate>=0.16.0",
    "encodec>=0.1.1",
]

COQUI_REQUIREMENTS = [
    "TTS>=0.11.0",
]

AENEAS_REQUIREMENTS = [
    "aeneas>=1.7.3",
]

WHISPER_REQUIREMENTS = [
    "openai-whisper>=20230314",
    "ffmpeg-python>=0.2.0",
]

def get_system_info():
    """Get system information for diagnostic purposes."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "pip_version": subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        ).stdout.strip(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    return info

def print_system_info():
    """Print system information."""
    info = get_system_info()
    print("System Information:")
    print("-" * 40)
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("-" * 40)

def create_requirements_file(components, output_file):
    """Create a requirements file for specified components."""
    requirements = BASIC_REQUIREMENTS.copy()
    
    # Add component-specific requirements
    if "bark" in components:
        requirements.extend(BARK_REQUIREMENTS)
    
    if "coqui" in components:
        requirements.extend(COQUI_REQUIREMENTS)
    
    if "aeneas" in components:
        requirements.extend(AENEAS_REQUIREMENTS)
    
    if "whisper" in components:
        requirements.extend(WHISPER_REQUIREMENTS)
    
    # Write requirements to file
    with open(output_file, "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"Requirements file created: {output_file}")
    return output_file

def install_requirements(requirements_file, use_pip=True, use_conda=False, verbose=False):
    """Install requirements using pip or conda."""
    print(f"Installing requirements from {requirements_file}...")
    
    if use_pip:
        cmd = [
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ]
        if verbose:
            print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=not verbose)
        if result.returncode != 0:
            print("Error installing requirements with pip:")
            print(result.stderr.decode() if not verbose else "")
            return False
    
    if use_conda:
        # Read requirements and install with conda
        with open(requirements_file, "r") as f:
            requirements = [line.strip() for line in f if line.strip()]
        
        for req in requirements:
            # Remove version constraints for conda
            package = req.split(">=")[0].split("==")[0].split("<")[0].strip()
            
            cmd = ["conda", "install", "-y", package]
            if verbose:
                print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=not verbose)
            if result.returncode != 0:
                print(f"Warning: Could not install {package} with conda")
    
    return True

def install_aeneas_dependencies():
    """Install system dependencies for aeneas."""
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing system dependencies for aeneas on Linux...")
        # Check if we're on Ubuntu/Debian
        try:
            subprocess.run(
                ["apt-get", "--version"], 
                capture_output=True, 
                check=False
            )
            subprocess.run(
                [
                    "sudo", "apt-get", "update"
                ],
                check=False
            )
            subprocess.run(
                [
                    "sudo", "apt-get", "install", "-y",
                    "espeak", "libespeak-dev", "ffmpeg", "libavcodec-extra"
                ],
                check=False
            )
        except:
            print("Could not install using apt-get. Please install espeak, libespeak-dev, and ffmpeg manually.")
    
    elif system == "darwin":  # macOS
        print("Installing system dependencies for aeneas on macOS...")
        try:
            # Check if brew is installed
            subprocess.run(["brew", "--version"], capture_output=True, check=True)
            subprocess.run(["brew", "install", "espeak", "ffmpeg"], check=False)
        except:
            print("Could not install using Homebrew. Please install espeak and ffmpeg manually.")
    
    elif system == "windows":
        print("On Windows, aeneas requires additional setup:")
        print("1. Download and install FFmpeg: https://ffmpeg.org/download.html")
        print("2. Download and install eSpeak: http://espeak.sourceforge.net/download.html")
        print("3. Add both to your PATH environment variable")
        print("\nAlternatively, consider using the Docker image for aeneas:")
        print("   https://hub.docker.com/r/readbeyond/aeneas/")
    
    else:
        print(f"Unknown system: {system}. Please install espeak and ffmpeg manually.")

def install_ffmpeg():
    """Make sure ffmpeg is installed for Whisper."""
    system = platform.system().lower()
    
    if system == "linux":
        try:
            subprocess.run(
                ["apt-get", "--version"], 
                capture_output=True, 
                check=False
            )
            print("Installing ffmpeg on Linux...")
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "ffmpeg"],
                check=False
            )
        except:
            print("Could not install ffmpeg using apt-get. Please install ffmpeg manually.")
    
    elif system == "darwin":  # macOS
        try:
            # Check if brew is installed
            subprocess.run(["brew", "--version"], capture_output=True, check=True)
            print("Installing ffmpeg on macOS...")
            subprocess.run(["brew", "install", "ffmpeg"], check=False)
        except:
            print("Could not install ffmpeg using Homebrew. Please install ffmpeg manually.")
    
    elif system == "windows":
        print("On Windows, please download and install FFmpeg manually:")
        print("1. Download FFmpeg: https://ffmpeg.org/download.html")
        print("2. Add it to your PATH environment variable")
    
    else:
        print(f"Unknown system: {system}. Please install ffmpeg manually.")

def test_requirements(components):
    """Test if requirements are properly installed."""
    all_passed = True
    
    # Test basic requirements
    try:
        import numpy
        import scipy
        import tqdm
        print("✓ Basic dependencies installed correctly")
    except ImportError as e:
        print(f"✗ Basic dependency error: {e}")
        all_passed = False
    
    # Test Bark dependencies
    if "bark" in components:
        try:
            import torch
            import transformers
            import accelerate
            print("✓ Bark dependencies installed correctly")
        except ImportError as e:
            print(f"✗ Bark dependency error: {e}")
            all_passed = False
    
    # Test Coqui dependencies
    if "coqui" in components:
        try:
            import TTS
            print("✓ Coqui TTS dependencies installed correctly")
        except ImportError as e:
            print(f"✗ Coqui TTS dependency error: {e}")
            all_passed = False
    
    # Test Aeneas dependencies
    if "aeneas" in components:
        try:
            import aeneas
            print("✓ Aeneas dependencies installed correctly")
            
            # Test if aeneas can actually run
            try:
                from aeneas.executetask import ExecuteTask
                from aeneas.task import Task
                print("✓ Aeneas functionality works")
            except Exception as e:
                print(f"✗ Aeneas functionality error: {e}")
                all_passed = False
                
        except ImportError as e:
            print(f"✗ Aeneas dependency error: {e}")
            all_passed = False
    
    # Test Whisper dependencies
    if "whisper" in components:
        try:
            import whisper
            print("✓ Whisper dependencies installed correctly")
        except ImportError as e:
            print(f"✗ Whisper dependency error: {e}")
            all_passed = False
    
    return all_passed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup requirements for narration generator"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["all", "bark", "coqui", "aeneas", "whisper"],
        default=["all"],
        help="Which TTS/alignment components to install (default: all)"
    )
    parser.add_argument(
        "--conda",
        action="store_true",
        help="Use conda to install packages (in addition to pip)"
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Skip installing system dependencies (espeak, ffmpeg)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output during installation"
    )
    parser.add_argument(
        "--requirements-only",
        action="store_true",
        help="Only generate requirements.txt file without installing"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test if requirements are properly installed"
    )
    
    return parser.parse_args()

def main():
    """Main function to set up requirements."""
    print("=" * 80)
    print("AI Video Teacher - Narration Generator Requirements Setup")
    print("=" * 80)
    
    args = parse_args()
    print_system_info()
    
    # Determine which components to install
    components = args.components
    if "all" in components:
        components = ["bark", "coqui", "aeneas", "whisper"]
    
    print(f"Components selected: {', '.join(components)}")
    
    # Create requirements file
    requirements_file = Path("narration_requirements.txt")
    create_requirements_file(components, requirements_file)
    
    if args.requirements_only:
        print("Requirements file generated. Exiting without installation.")
        return 0
    
    # Install system dependencies if needed
    if not args.skip_system_deps:
        if "aeneas" in components:
            install_aeneas_dependencies()
        
        if "whisper" in components:
            install_ffmpeg()
    
    # Install Python requirements
    if not install_requirements(
        requirements_file, 
        use_conda=args.conda,
        verbose=args.verbose
    ):
        print("Failed to install requirements.")
        return 1
    
    # Test the installation
    if args.test or True:  # Always test
        print("\nTesting installation...")
        if test_requirements(components):
            print("\nAll requirements installed successfully!")
        else:
            print("\nSome requirements couldn't be installed. See errors above.")
            return 1
    
    print("\nSetup completed!")
    print("You can now use the narration generator with:")
    print("python examples/demo_generate_narration.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 