#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Installation script for Narration Generator dependencies.
This script installs the required dependencies for the narration generator component.
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path

# Define colors for terminal output
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

def print_section(title):
    """Print a section title"""
    print(f"\n{BLUE}{BOLD}==== {title} ===={ENDC}\n")

def print_status(message, status_type="info"):
    """Print a status message with appropriate color"""
    if status_type == "info":
        color = BLUE
    elif status_type == "success":
        color = GREEN
    elif status_type == "warning":
        color = YELLOW
    elif status_type == "error":
        color = RED
    else:
        color = ""
        
    print(f"{color}{message}{ENDC}")

def run_command(command, verbose=True):
    """Run a shell command and return the result"""
    try:
        if verbose:
            print_status(f"Running: {command}")
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print_status(f"Command failed with code {process.returncode}", "error")
            print_status(f"Error: {stderr}", "error")
            return False
            
        return True
    except Exception as e:
        print_status(f"Exception while running command: {e}", "error")
        return False

def check_python_version():
    """Check if the Python version is suitable"""
    print_section("Checking Python Version")
    
    major, minor, _ = sys.version_info
    if major < 3 or (major == 3 and minor < 8):
        print_status(f"Python {major}.{minor} detected. Version 3.8 or higher is required.", "error")
        return False
        
    print_status(f"Python {major}.{minor} detected. ✓", "success")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed and install it if necessary"""
    print_section("Checking FFmpeg Installation")
    
    # Check if FFmpeg is installed
    if run_command("ffmpeg -version", verbose=False):
        print_status("FFmpeg is already installed. ✓", "success")
        return True
        
    print_status("FFmpeg not found. Attempting to install...", "warning")
    
    # Install FFmpeg based on the platform
    system = platform.system().lower()
    
    if system == "linux":
        if run_command("apt-get -y update && apt-get -y install ffmpeg"):
            print_status("FFmpeg installed successfully. ✓", "success")
            return True
    elif system == "darwin":  # macOS
        if run_command("brew install ffmpeg"):
            print_status("FFmpeg installed successfully. ✓", "success")
            return True
    elif system == "windows":
        print_status("Please install FFmpeg manually on Windows:", "warning")
        print_status("1. Download from https://ffmpeg.org/download.html", "warning")
        print_status("2. Add it to your PATH", "warning")
        # Return True as we can't install it automatically but don't want to block the installation
        return True
        
    print_status("Could not install FFmpeg automatically. Please install it manually.", "error")
    return False

def install_core_dependencies():
    """Install core Python dependencies"""
    print_section("Installing Core Dependencies")
    
    dependencies = [
        "numpy",
        "torch",
        "soundfile",
        "requests",
        "tqdm",
        "scikit-learn",
        "textstat",
        "scipy"
    ]
    
    # Check if CUDA is available for PyTorch
    if torch_with_cuda:
        pip_command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            
        if not run_command(pip_command):
            print_status("Failed to install PyTorch with CUDA support.", "error")
            
        # Remove torch from dependencies as we installed it separately
        dependencies = [dep for dep in dependencies if dep != "torch"]
        
    # Install core dependencies
    pip_command = f"{sys.executable} -m pip install {' '.join(dependencies)}"
    if not run_command(pip_command):
        print_status("Failed to install core dependencies.", "error")
        return False
        
    print_status("Core dependencies installed successfully. ✓", "success")
    return True

def install_bark_tts():
    """Install Bark TTS dependencies"""
    print_section("Installing Bark TTS")
    
    dependencies = [
        "git+https://github.com/suno-ai/bark.git",
        "transformers>=4.31.0",
        "encodec",
    ]
    
    pip_command = f"{sys.executable} -m pip install {' '.join(dependencies)}"
    if not run_command(pip_command):
        print_status("Failed to install Bark TTS.", "error")
        return False
        
    print_status("Bark TTS installed successfully. ✓", "success")
    
    # Download models
    print_status("Downloading Bark models (this may take a while)...", "info")
    try:
        import os
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"  # Use small models for faster download
        
        import bark
        from bark import preload_models
        preload_models()
        
        print_status("Bark models downloaded successfully. ✓", "success")
        return True
    except Exception as e:
        print_status(f"Failed to download Bark models: {e}", "error")
        print_status("Models will be downloaded on first use.", "warning")
        return True  # Return True anyway as this is not a critical error

def install_coqui_tts():
    """Install Coqui TTS dependencies"""
    print_section("Installing Coqui TTS")
    
    dependencies = [
        "TTS",
    ]
    
    pip_command = f"{sys.executable} -m pip install {' '.join(dependencies)}"
    if not run_command(pip_command):
        print_status("Failed to install Coqui TTS.", "error")
        return False
        
    print_status("Coqui TTS installed successfully. ✓", "success")
    return True

def install_aeneas():
    """Install Aeneas dependencies"""
    print_section("Installing Aeneas")
    
    # Platform-specific installation for Aeneas (it has complex dependencies)
    system = platform.system().lower()
    
    if system == "linux":
        # Install espeak and other dependencies
        apt_command = "apt-get -y update && apt-get -y install espeak libespeak-dev python3-dev python3-pip build-essential"
        if not run_command(apt_command):
            print_status("Failed to install Aeneas system dependencies.", "error")
            return False
    elif system == "darwin":  # macOS
        # Install espeak using brew
        if not run_command("brew install espeak"):
            print_status("Failed to install espeak on macOS.", "error")
            return False
    elif system == "windows":
        print_status("On Windows, Aeneas installation requires additional steps:", "warning")
        print_status("1. Download and install espeak from http://espeak.sourceforge.net/download.html", "warning")
        print_status("2. Add espeak to your PATH", "warning")
        print_status("3. Install the Microsoft Visual C++ Build Tools", "warning")
        print_status("Attempting to install Aeneas anyway...", "warning")
    
    # Install Aeneas from pip
    pip_command = f"{sys.executable} -m pip install numpy aeneas"
    if not run_command(pip_command):
        print_status("Failed to install Aeneas. Please try manual installation.", "error")
        return False
    
    # Check if Aeneas works
    check_command = f"{sys.executable} -c \"import aeneas; print('Aeneas works!');\""
    if not run_command(check_command):
        print_status("Aeneas installed but not working correctly. You may need to install it manually.", "warning")
        return True  # Return True to continue with other installations
    
    print_status("Aeneas installed successfully. ✓", "success")
    return True

def install_gentle():
    """Install or configure Gentle"""
    print_section("Configuring Gentle")
    
    print_status("Gentle requires a separate server installation.", "info")
    print_status("To use Gentle aligner, follow these steps:", "info")
    print_status("1. Clone Gentle from https://github.com/lowerquality/gentle", "info")
    print_status("2. Follow the installation instructions in the Gentle repository", "info")
    print_status("3. Run the Gentle server with: python3 gentle/serve.py", "info")
    print_status("4. The narration generator will connect to the server at http://localhost:8765", "info")
    
    print_status("Skipping Gentle installation as it requires manual setup. ✓", "warning")
    return True

def install_whisper():
    """Install OpenAI Whisper dependencies"""
    print_section("Installing Whisper")
    
    dependencies = [
        "openai-whisper",
        "ffmpeg-python",
    ]
    
    pip_command = f"{sys.executable} -m pip install {' '.join(dependencies)}"
    if not run_command(pip_command):
        print_status("Failed to install Whisper.", "error")
        return False
        
    print_status("Whisper installed successfully. ✓", "success")
    return True

def create_requirements_file(output_dir="."):
    """Create a requirements.txt file with all the dependencies"""
    print_section("Creating Requirements File")
    
    requirements = [
        "# Core dependencies",
        "numpy",
        "torch",
        "soundfile",
        "requests",
        "tqdm",
        "scikit-learn",
        "textstat",
        "scipy",
        "",
        "# Bark TTS",
        "git+https://github.com/suno-ai/bark.git",
        "transformers>=4.31.0",
        "encodec",
        "",
        "# Coqui TTS",
        "TTS",
        "",
        "# Whisper",
        "openai-whisper",
        "ffmpeg-python",
        "",
        "# Aeneas",
        "aeneas",
        "",
        "# Note: Gentle requires manual installation",
        "# See: https://github.com/lowerquality/gentle",
    ]
    
    output_path = os.path.join(output_dir, "requirements.txt")
    
    try:
        with open(output_path, "w") as f:
            f.write("\n".join(requirements))
            
        print_status(f"Requirements file created at {output_path}. ✓", "success")
        return True
    except Exception as e:
        print_status(f"Failed to create requirements file: {e}", "error")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install dependencies for narration generator")
    parser.add_argument("--cuda", action="store_true", help="Install with CUDA support for faster processing")
    parser.add_argument("--minimal", action="store_true", help="Install only core dependencies without TTS engines")
    parser.add_argument("--tts", choices=["all", "bark", "coqui"], default="all", help="TTS engines to install")
    parser.add_argument("--aligner", choices=["all", "aeneas", "whisper", "gentle"], default="aeneas", help="Aligner to install")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for requirements.txt")
    
    args = parser.parse_args()
    
    # Set global flag for CUDA support
    global torch_with_cuda
    torch_with_cuda = args.cuda
    
    print_section("AI Video Teacher - Narration Generator Setup")
    print_status("Starting installation of narration generator dependencies...", "info")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check and install FFmpeg
    check_ffmpeg()
    
    # Install core dependencies
    if not install_core_dependencies():
        print_status("Failed to install core dependencies. Aborting.", "error")
        sys.exit(1)
    
    # Install TTS engines
    if not args.minimal:
        if args.tts in ["all", "bark"]:
            install_bark_tts()
        
        if args.tts in ["all", "coqui"]:
            install_coqui_tts()
    
    # Install aligners
    if args.aligner == "all":
        install_aeneas()
        install_whisper()
        install_gentle()
    elif args.aligner == "aeneas":
        install_aeneas()
    elif args.aligner == "whisper":
        install_whisper()
    elif args.aligner == "gentle":
        install_gentle()
    
    # Create requirements.txt
    create_requirements_file(args.output_dir)
    
    print_section("Installation Complete")
    print_status("Narration generator dependencies installed successfully!", "success")
    print_status("You can now use the narration generator with:", "info")
    print_status("python -m src.narration_generator.core --help", "info")

if __name__ == "__main__":
    # Initialize global variables
    torch_with_cuda = False
    
    try:
        main()
    except KeyboardInterrupt:
        print_status("\nInstallation interrupted by user.", "warning")
        sys.exit(1)
    except Exception as e:
        print_status(f"\nUnexpected error: {e}", "error")
        sys.exit(1) 