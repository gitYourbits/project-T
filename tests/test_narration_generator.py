#!/usr/bin/env python
"""
Tests for the NarrationGenerator module.

This script tests the basic functionality of the narration generator.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.narration_generator.core import NarrationGenerator
from src.narration_generator.tts_engines import TTSEngine
from src.narration_generator.aligners import SpeechAligner


class MockTTSEngine(TTSEngine):
    """Mock TTS engine for testing."""
    
    def __init__(self, name="mock_tts", supports_emotions=True):
        self._name = name
        self._supports_emotions = supports_emotions
    
    def generate_speech(self, text, options=None):
        # Return a mock audio array and sample rate
        import numpy as np
        audio = np.zeros(int(len(text) * 1000))  # 1000 samples per character
        sample_rate = 22050
        return audio, sample_rate
    
    def save_to_file(self, audio_data, output_path):
        # Mock saving audio to file
        audio_array, sample_rate = audio_data
        
        # Create directories
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save a dummy audio file (just write the length to a file)
        with open(output_path, 'w') as f:
            f.write(f"Mock audio file: {len(audio_array)} samples at {sample_rate}Hz")
        
        return output_path
    
    @property
    def supported_languages(self):
        return ["en"]
    
    @property
    def supports_emotions(self):
        return self._supports_emotions
    
    @property
    def name(self):
        return self._name


class MockSpeechAligner(SpeechAligner):
    """Mock speech aligner for testing."""
    
    def __init__(self, name="mock_aligner"):
        self._name = name
    
    def align(self, audio_path, text):
        # Generate mock alignment data
        words = text.split()
        alignment = []
        current_time = 0.0
        
        for word in words:
            # Simulate alignment with roughly 0.3 seconds per word
            duration = 0.1 + len(word) * 0.05  # Longer words take more time
            
            alignment.append({
                "start": current_time,
                "end": current_time + duration,
                "text": word
            })
            
            current_time += duration
        
        return alignment
    
    @property
    def name(self):
        return self._name


class TestNarrationGenerator(unittest.TestCase):
    """Test cases for NarrationGenerator."""
    
    def setUp(self):
        """Set up for tests."""
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a mock TTS engine and speech aligner
        self.mock_tts = MockTTSEngine()
        self.mock_aligner = MockSpeechAligner()
        
        # Create a narration generator with mocks
        self.generator = NarrationGenerator(
            tts_engine=self.mock_tts,
            speech_aligner=self.mock_aligner,
            output_dir=str(self.test_dir)
        )
        
        # Sample scene content
        self.scene_content = {
            "scene": 1,
            "title": "Introduction to Testing",
            "scene_type": "introduction",
            "natural_content": "This is a test narration for the introduction scene. It includes some basic content to test the narration generator."
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # In a real test, you might want to remove test files
        # import shutil
        # shutil.rmtree(self.test_dir)
        pass
    
    def test_prepare_text_for_narration(self):
        """Test text preparation for narration."""
        # Test with various markers and formatting
        test_text = """# Introduction
        This is a test with various markers.
        
        @@diagram: A test diagram
        
        • Bullet point 1
        • Bullet point 2
        
        [Visual: A test visual cue]
        """
        
        expected = "This is a test with various markers. Bullet point 1 Bullet point 2"
        
        result = self.generator._prepare_text_for_narration(test_text)
        self.assertEqual(result, expected)
    
    def test_enhance_voice_options(self):
        """Test voice options enhancement."""
        # Test with no base options
        base_options = {}
        result = self.generator._enhance_voice_options(base_options, self.scene_content)
        self.assertIn("emotion", result)
        self.assertEqual(result["emotion"], "excited")  # introduction scene type
        
        # Test with existing options
        base_options = {"emotion": "happy", "speaking_rate": 1.0}
        result = self.generator._enhance_voice_options(base_options, self.scene_content)
        self.assertEqual(result["emotion"], "happy")  # should not override
        self.assertEqual(result["speaking_rate"], 1.0)
    
    def test_generate_narration(self):
        """Test narration generation for a single scene."""
        result = self.generator.generate_narration(self.scene_content, output_prefix="test")
        
        # Check result structure
        self.assertEqual(result["scene_id"], 1)
        self.assertEqual(result["status"], "success")
        self.assertIn("audio_path", result)
        self.assertIn("alignment_path", result)
        self.assertIn("duration", result)
        
        # Check that files were created
        audio_path = Path(result["audio_path"])
        alignment_path = Path(result["alignment_path"])
        
        self.assertTrue(audio_path.exists())
        self.assertTrue(alignment_path.exists())
    
    def test_generate_narrations_for_script(self):
        """Test narration generation for a complete script."""
        # Create a mock script with two scenes
        script_data = {
            "metadata": {
                "total_scenes": 2
            },
            "scenes": [
                self.scene_content,
                {
                    "scene": 2,
                    "title": "Main Content",
                    "scene_type": "main_content",
                    "natural_content": "This is the main content section. It contains more information about the topic."
                }
            ]
        }
        
        result = self.generator.generate_narrations_for_script(script_data, output_prefix="test_script")
        
        # Check result structure
        self.assertEqual(result["metadata"]["total_scenes"], 2)
        self.assertEqual(result["metadata"]["success_count"], 2)
        self.assertEqual(len(result["narrations"]), 2)
        
        # Check summary file
        summary_path = self.test_dir / "test_script_narration_summary.json"
        self.assertTrue(summary_path.exists())
    
    def test_empty_content(self):
        """Test handling of empty content."""
        empty_scene = {
            "scene": 3,
            "title": "Empty Scene",
            "scene_type": "main_content",
            "natural_content": ""
        }
        
        result = self.generator.generate_narration(empty_scene)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "no_content")
    
    def test_error_handling(self):
        """Test error handling."""
        # Create a scene that will cause an error
        with patch.object(self.mock_tts, 'generate_speech', side_effect=RuntimeError("Test error")):
            result = self.generator.generate_narration(self.scene_content)
            self.assertEqual(result["status"], "error")
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main() 