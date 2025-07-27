#!/usr/bin/env python3
"""
Caption Generator for Video Content
Adds customizable captions/subtitles to video segments.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from tqdm import tqdm

class CaptionGenerator:
    def __init__(self, config_path: str = "config.json", gpu_manager=None):
        """Initialize Caption Generator with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.gpu_manager = gpu_manager
        
        # Default caption settings
        self.default_caption_settings = {
            "enabled": True,
            "position": "bottom",  # top, bottom, center
            "font": "Arial-Bold",
            "fontsize": 48,
            "color": "white",
            "stroke_color": "black",
            "stroke_width": 2,
            "background": True,
            "background_color": "black",
            "background_opacity": 0.7,
            "margin_bottom": 50,
            "margin_top": 50,
            "max_width": 0.8,  # 80% of video width
            "animation": "fade",  # none, fade, slide
            "duration_buffer": 0.1  # Extra time to show caption
        }
    
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
    
    def add_captions_to_videos(self, script_path: str) -> bool:
        """
        Add captions to all video segments based on script.
        
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
            caption_settings = script.get('caption_settings', self.default_caption_settings)
            
            # Skip if captions are disabled
            if not caption_settings.get('enabled', True):
                self.logger.info("Captions disabled in script settings")
                return True
            
            self.logger.info(f"Adding captions to {len(segments)} video segments...")
            
            # Ensure output directory exists
            Path("temp/captioned").mkdir(parents=True, exist_ok=True)
            
            # Process each segment
            for segment in tqdm(segments, desc="Adding captions"):
                success = self.add_caption_to_segment(segment, caption_settings, script)
                if not success:
                    self.logger.error(f"Failed to add caption to segment {segment['id']}")
                    return False
            
            self.logger.info("All captions added successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding captions: {e}")
            return False
    
    def add_caption_to_segment(self, segment: Dict, caption_settings: Dict, script: Dict) -> bool:
        """
        Add caption to a single video segment.
        
        Args:
            segment: Segment configuration dictionary
            caption_settings: Caption styling settings
            script: Complete script configuration
            
        Returns:
            bool: Success status
        """
        try:
            segment_id = segment['id']
            text = segment['text']
            
            # Input and output paths
            input_path = Path(f"temp/composed/segment_{segment_id}.mp4")
            output_path = Path(f"temp/captioned/segment_{segment_id}.mp4")
            
            # Check if input exists
            if not input_path.exists():
                self.logger.error(f"Input video not found: {input_path}")
                return False
            
            # Load video clip
            video_clip = VideoFileClip(str(input_path))
            
            # Get video dimensions
            video_width, video_height = video_clip.size
            
            # Create caption clip
            caption_clip = self.create_caption_clip(
                text, 
                video_clip.duration, 
                video_width, 
                video_height, 
                caption_settings
            )
            
            # Compose video with caption
            if caption_clip:
                final_video = CompositeVideoClip([video_clip, caption_clip])
            else:
                # Fallback to original video if caption creation fails
                final_video = video_clip
            
            # Write output video
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp/temp_caption_audio.m4a',
                remove_temp=True,
                logger=None
            )
            
            # Clean up
            video_clip.close()
            if caption_clip:
                caption_clip.close()
            final_video.close()
            
            self.logger.info(f"Added caption to segment {segment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding caption to segment {segment_id}: {e}")
            return False
    
    def create_caption_clip(self, text: str, duration: float, video_width: int, 
                           video_height: int, settings: Dict) -> Optional[TextClip]:
        """
        Create a caption text clip with styling.
        
        Args:
            text: Caption text
            duration: Duration of the caption
            video_width: Width of the video
            video_height: Height of the video
            settings: Caption styling settings
            
        Returns:
            TextClip or CompositeVideoClip with caption
        """
        try:
            # Split text into lines if too long
            max_chars_per_line = 60  # Adjust based on font size
            lines = self.split_text_to_lines(text, max_chars_per_line)
            
            # Create text clip
            text_clip = TextClip(
                '\n'.join(lines),
                fontsize=settings.get('fontsize', 48),
                font=settings.get('font', 'Arial-Bold'),
                color=settings.get('color', 'white'),
                stroke_color=settings.get('stroke_color', 'black'),
                stroke_width=settings.get('stroke_width', 2),
                method='caption',
                size=(int(video_width * settings.get('max_width', 0.8)), None)
            ).set_duration(duration + settings.get('duration_buffer', 0.1))
            
            # Position the text
            position = self.get_text_position(settings.get('position', 'bottom'), 
                                            video_width, video_height, 
                                            text_clip.size, settings)
            text_clip = text_clip.set_position(position)
            
            # Add background if enabled
            if settings.get('background', True):
                return self.add_background_to_text(text_clip, settings, video_width, video_height)
            
            # Add animation if specified
            animation = settings.get('animation', 'none')
            if animation == 'fade':
                text_clip = text_clip.crossfadein(0.3).crossfadeout(0.3)
            elif animation == 'slide':
                text_clip = self.add_slide_animation(text_clip, settings, video_width, video_height)
            
            return text_clip
            
        except Exception as e:
            self.logger.error(f"Error creating caption clip: {e}")
            return None
    
    def split_text_to_lines(self, text: str, max_chars_per_line: int) -> List[str]:
        """Split text into multiple lines for better readability."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def get_text_position(self, position: str, video_width: int, video_height: int, 
                         text_size: Tuple[int, int], settings: Dict) -> Tuple[str, int]:
        """Calculate text position based on settings."""
        text_width, text_height = text_size
        
        if position == 'bottom':
            x = 'center'
            y = video_height - text_height - settings.get('margin_bottom', 50)
        elif position == 'top':
            x = 'center'
            y = settings.get('margin_top', 50)
        elif position == 'center':
            x = 'center'
            y = 'center'
        else:
            # Default to bottom
            x = 'center'
            y = video_height - text_height - settings.get('margin_bottom', 50)
        
        return (x, y)
    
    def add_background_to_text(self, text_clip: TextClip, settings: Dict, 
                              video_width: int, video_height: int) -> CompositeVideoClip:
        """Add a background box behind the text."""
        try:
            # Get text dimensions
            text_width, text_height = text_clip.size
            
            # Create background with padding
            padding = 20
            bg_width = text_width + (padding * 2)
            bg_height = text_height + (padding * 2)
            
            # Create background color clip
            bg_color = settings.get('background_color', 'black')
            bg_opacity = settings.get('background_opacity', 0.7)
            
            background = ColorClip(
                size=(bg_width, bg_height),
                color=bg_color
            ).set_duration(text_clip.duration).set_opacity(bg_opacity)
            
            # Position background behind text
            text_position = text_clip.pos
            if callable(text_position):
                # Handle dynamic positioning
                bg_position = ('center', text_position(0)[1] - padding)
            else:
                if text_position[0] == 'center':
                    bg_x = 'center'
                else:
                    bg_x = text_position[0] - padding
                bg_y = text_position[1] - padding
                bg_position = (bg_x, bg_y)
            
            background = background.set_position(bg_position)
            
            # Composite background and text
            return CompositeVideoClip([background, text_clip])
            
        except Exception as e:
            self.logger.error(f"Error adding background to text: {e}")
            return text_clip
    
    def add_slide_animation(self, text_clip: TextClip, settings: Dict, 
                           video_width: int, video_height: int) -> TextClip:
        """Add slide-in animation to text."""
        try:
            position = settings.get('position', 'bottom')
            
            if position == 'bottom':
                # Slide up from bottom
                start_y = video_height
                end_y = video_height - text_clip.size[1] - settings.get('margin_bottom', 50)
            elif position == 'top':
                # Slide down from top
                start_y = -text_clip.size[1]
                end_y = settings.get('margin_top', 50)
            else:
                # No slide animation for center
                return text_clip
            
            # Create position function for sliding
            def pos_func(t):
                if t < 0.5:  # Slide in during first 0.5 seconds
                    progress = t / 0.5
                    y = start_y + (end_y - start_y) * progress
                else:
                    y = end_y
                return ('center', y)
            
            return text_clip.set_position(pos_func)
            
        except Exception as e:
            self.logger.error(f"Error adding slide animation: {e}")
            return text_clip
    
    def update_video_paths_for_captions(self, script_path: str) -> bool:
        """Update video assembly to use captioned videos instead of composed ones."""
        try:
            # This would be called by the video assembler to use captioned videos
            # Check if captioned videos exist
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            segments = script['segments']
            captioned_dir = Path("temp/captioned")
            
            if not captioned_dir.exists():
                return False
            
            # Check if all captioned videos exist
            for segment in segments:
                captioned_path = captioned_dir / f"segment_{segment['id']}.mp4"
                if not captioned_path.exists():
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking captioned videos: {e}")
            return False

def main():
    """Main function for testing caption generation."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python caption_generator.py <script_path>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    generator = CaptionGenerator()
    
    success = generator.add_captions_to_videos(script_path)
    if success:
        print("✅ Captions added successfully!")
    else:
        print("❌ Failed to add captions")
        sys.exit(1)

if __name__ == "__main__":
    main() 