#!/usr/bin/env python
"""
AI Video Teacher - Narration Generation Demo

This script demonstrates how to use the narration generator with a sample script file.
It creates a minimal example script and processes it with different TTS engines and aligners.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Available TTS engines
TTS_ENGINES = ["bark", "coqui"]
ALIGNERS = ["aeneas", "whisper", "none"]

# Sample script content
SAMPLE_SCRIPT = """# Introduction to Python Programming

## Scene 1: Introduction to Programming
In this lesson, we'll learn about Python programming language. Python is a high-level, interpreted programming language that is known for its readability and simplicity.

* Python was created by Guido van Rossum in the late 1980s
* It is named after the British comedy group Monty Python
* Python is often used for web development, data analysis, artificial intelligence, and automation

## Scene 2: Your First Python Program
Let's write our first Python program. The traditional first program in any language is "Hello, World!".

```python
print("Hello, World!")
```

When you run this program, it will display the text "Hello, World!" in the console.

## Scene 3: Basic Python Syntax
Python uses indentation to define code blocks, unlike other languages that use braces.

```python
if True:
    print("This is indented")
    if True:
        print("This is further indented")
print("This is not indented")
```

This makes Python code very readable and forces a consistent coding style.
"""

def create_sample_script(output_dir):
    """Create a sample script file for demonstration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = output_dir / "sample_script.md"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_SCRIPT)
    
    print(f"Created sample script: {script_path}")
    return script_path

def run_narration_generation(script_path, tts_engine, aligner, output_dir):
    """Run the narration generation script with given options."""
    import subprocess
    
    # Form the command
    cmd = [
        sys.executable,
        "examples/generate_narration.py",
        "--script", str(script_path),
        "--output", str(output_dir),
        "--tts-engine", tts_engine,
        "--aligner", aligner,
        "--verbose"
    ]
    
    # Add specific options based on TTS engine
    if tts_engine == "bark":
        cmd.extend(["--voice", "v2/en_speaker_6"])
    elif tts_engine == "coqui":
        cmd.extend(["--voice", "p230"])
    
    print(f"\nRunning narration generation with {tts_engine} and {aligner}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Narration generation completed successfully.")
        return True
    else:
        print(f"Narration generation failed with exit code {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        return False

def check_requirements():
    """Check if required libraries are installed."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo for narration generation system"
    )
    parser.add_argument(
        "--tts-engine",
        choices=TTS_ENGINES,
        default="bark",
        help="TTS engine to use (default: bark)"
    )
    parser.add_argument(
        "--aligner",
        choices=ALIGNERS,
        default="aeneas",
        help="Speech aligner to use (default: aeneas)"
    )
    parser.add_argument(
        "--output-dir",
        default="demo_output",
        help="Output directory for generated files (default: demo_output)"
    )
    parser.add_argument(
        "--sample-script",
        help="Path to a custom sample script (default: create a built-in sample)"
    )
    
    return parser.parse_args()

def main():
    """Main function for the narration generation demo."""
    print("=" * 80)
    print("AI Video Teacher - Narration Generation Demo")
    print("=" * 80)
    
    args = parse_args()
    
    # Check dependencies
    if not check_requirements():
        return 1
    
    # Create or use provided sample script
    if args.sample_script:
        script_path = Path(args.sample_script)
        if not script_path.exists():
            print(f"Error: Sample script {script_path} does not exist")
            return 1
    else:
        script_path = create_sample_script(args.output_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.tts_engine}_{args.aligner}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run narration generation
    success = run_narration_generation(
        script_path, 
        args.tts_engine, 
        args.aligner, 
        output_dir
    )
    
    if success:
        # Show the output files
        print("\nGenerated files:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(str(output_dir), '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
        
        # Load and display summary
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            
            print("\nNarration Summary:")
            print(f"Total scenes: {summary.get('num_scenes', 0)}")
            print(f"Generated narrations: {summary.get('num_narrations', 0)}")
            print(f"Failed narrations: {summary.get('num_failed', 0)}")
            
            # Show how to use the generated files
            print("\nTo listen to the generated narrations:")
            for i, narration in enumerate(summary.get("narrations", [])):
                if "audio_file" in narration:
                    print(f"  Scene {i+1}: {narration['audio_file']}")
        
        print("\nDemo completed successfully!")
        return 0
    else:
        print("\nDemo failed. Please check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 