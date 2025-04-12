#!/usr/bin/env python
"""
Simplified demo script for narration generation.

This script demonstrates the narration generation workflow without requiring
external dependencies like Bark TTS or Aeneas.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.script_parser.core import ScriptParser
from tests.test_narration_generator import MockTTSEngine, MockSpeechAligner
from src.narration_generator.core import NarrationGenerator


def main():
    """Main function that demonstrates the narration generation process."""
    # Step 1: Set up paths
    script_path = Path("examples/sample_script.md")
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== AI Video Teacher Narration Demo ===")
    logger.info(f"Processing script: {script_path}")
    
    try:
        # Step 2: Parse the script
        logger.info("Step 1: Parsing script...")
        script_parser = ScriptParser()
        script_text = script_path.read_text(encoding="utf-8")
        parsed_script = script_parser.parse_script(script_text)
        
        # Save parsed script for reference
        parsed_script_path = output_dir / "parsed_script.json"
        with open(parsed_script_path, "w", encoding="utf-8") as f:
            json.dump(parsed_script, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Parsed script saved to {parsed_script_path}")
        
        # Print some information about the parsed script
        logger.info(f"Script contains {len(parsed_script['scenes'])} scenes")
        
        # Step 3: Initialize mock engines
        logger.info("Step 2: Creating narration generator...")
        mock_tts = MockTTSEngine(name="Demo TTS", supports_emotions=True)
        mock_aligner = MockSpeechAligner(name="Demo Aligner")
        
        # Step 4: Initialize Narration Generator
        narration_generator = NarrationGenerator(
            tts_engine=mock_tts,
            speech_aligner=mock_aligner,
            output_dir=output_dir / "narrations"
        )
        
        # Step 5: Generate narrations
        logger.info("Step 3: Generating narrations for all scenes...")
        
        voice_options = {
            "emotion": "neutral",
            "voice_preset": "demo_voice"
        }
        
        narration_results = narration_generator.generate_narrations_for_script(
            parsed_script,
            output_prefix="demo",
            voice_options=voice_options
        )
        
        # Step 6: Save and display results
        summary_path = output_dir / "narration_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(narration_results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        success_count = narration_results["metadata"]["success_count"]
        total_scenes = narration_results["metadata"]["total_scenes"]
        total_duration = narration_results["metadata"]["total_duration"]
        
        logger.info(f"=== Narration Generation Complete ===")
        logger.info(f"Successfully processed {success_count}/{total_scenes} scenes")
        logger.info(f"Total narration duration: {total_duration:.2f} seconds")
        logger.info(f"Results saved to {summary_path}")
        
        # Display some sample alignment data
        logger.info("\n=== Sample Scene Narration ===")
        if narration_results["narrations"] and narration_results["narrations"][0]["status"] == "success":
            sample = narration_results["narrations"][0]
            logger.info(f"Scene: {sample['title']}")
            logger.info(f"Duration: {sample['duration']:.2f} seconds")
            logger.info(f"Word count: {sample['word_count']} words")
            
            # Show the file paths
            logger.info(f"Audio file: {sample['audio_path']}")
            logger.info(f"Alignment file: {sample['alignment_path']}")
            
            # Load and display a snippet of the alignment data
            with open(sample['alignment_path'], 'r') as f:
                alignment = json.load(f)
                
            logger.info("\n=== Timing Information (first 5 entries) ===")
            for i, entry in enumerate(alignment[:5]):
                logger.info(f"{entry['start']:.2f}s - {entry['end']:.2f}s: \"{entry['text']}\"")
            
            if len(alignment) > 5:
                logger.info(f"... and {len(alignment) - 5} more entries")
        
        logger.info("\n=== Demo Completed Successfully ===")
        logger.info(f"All output files are in: {output_dir}")
            
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 