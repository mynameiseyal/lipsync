#!/usr/bin/env python3
"""
Simple TTS Generator using gTTS
Generates audio files for each text segment using Google Text-to-Speech.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List
from gtts import gTTS
from pydub import AudioSegment
from tqdm import tqdm

class TTSGenerator:
    def __init__(self, config_path: str = "config.json"):
        """Initialize TTS Generator with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_audio_segments(self, script_path: str) -> bool:
        """
        Generate audio files for all segments in the script using gTTS.
        
        Args:
            script_path: Path to the script JSON file
            
        Returns:
            bool: Success status
        """
        try:
            # Load script
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            segments = script['segments']
            audio_settings = script.get('audio_settings', {})
            
            self.logger.info(f"Generating audio for {len(segments)} segments using gTTS...")
            
            # Create output directory
            audio_dir = Path("temp/audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each segment
            for segment in tqdm(segments, desc="Generating TTS audio"):
                success = self.generate_single_segment(segment, audio_settings)
                if not success:
                    self.logger.error(f"Failed to generate audio for segment {segment['id']}")
                    return False
            
            self.logger.info("All audio segments generated successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating audio segments: {e}")
            return False
    
    def generate_single_segment(self, segment: Dict, audio_settings: Dict) -> bool:
        """
        Generate audio for a single segment using gTTS.
        
        Args:
            segment: Segment configuration dictionary
            audio_settings: Audio processing settings
            
        Returns:
            bool: Success status
        """
        try:
            segment_id = segment['id']
            text = segment['text']
            
            # Generate using gTTS
            self.logger.info(f"Generating audio for segment {segment_id}...")
            tts = gTTS(text=text, lang='en', slow=False)
            temp_path = f"temp/audio/segment_{segment_id}_temp.mp3"
            output_path = f"temp/audio/segment_{segment_id}.wav"
            
            tts.save(temp_path)
            
            # Convert to WAV format
            audio = AudioSegment.from_mp3(temp_path)
            
            # Apply basic processing
            if audio_settings.get('normalize', True):
                # Basic normalization
                audio = audio.normalize()
            
            # Ensure consistent sample rate
            sample_rate = audio_settings.get('sample_rate', 22050)
            audio = audio.set_frame_rate(sample_rate)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            # Clean up temp file
            os.remove(temp_path)
            
            self.logger.info(f"Generated audio for segment {segment_id}: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating segment audio: {e}")
            return False
    
    def get_audio_duration(self, segment_id: int) -> float:
        """
        Get duration of generated audio segment.
        
        Args:
            segment_id: ID of the segment
            
        Returns:
            Duration in seconds
        """
        try:
            audio_path = f"temp/audio/segment_{segment_id}.wav"
            audio = AudioSegment.from_wav(audio_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            self.logger.error(f"Could not get duration for segment {segment_id}: {e}")
            return 0.0


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate TTS audio for lip-sync video')
    parser.add_argument('--script', default='input/script.json', 
                       help='Path to script JSON file')
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TTSGenerator(args.config)
    
    # Generate audio segments
    success = generator.generate_audio_segments(args.script)
    
    if success:
        print("✅ Audio generation completed successfully!")
    else:
        print("❌ Audio generation failed!")
        exit(1)


if __name__ == "__main__":
    main() 