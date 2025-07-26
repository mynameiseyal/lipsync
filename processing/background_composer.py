#!/usr/bin/env python3
"""
Background Composer for Multi-Speaker Video Generation
Combines lip-synced videos with custom background images.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, vfx
from PIL import Image, ImageFilter, ImageEnhance
from rembg import remove
from tqdm import tqdm

class BackgroundComposer:
    def __init__(self, config_path: str = "config.json"):
        """Initialize Background Composer with configuration."""
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
    
    def compose_all_videos(self, script_path: str) -> bool:
        """
        Compose all videos with their respective backgrounds.
        
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
            
            self.logger.info(f"Composing backgrounds for {len(segments)} segments...")
            
            # Create output directory
            composed_dir = Path("temp/composed")
            composed_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each segment
            for segment in tqdm(segments, desc="Composing backgrounds"):
                success = self.compose_single_video(segment, script)
                if not success:
                    self.logger.error(f"Failed to compose background for segment {segment['id']}")
                    return False
            
            self.logger.info("All background compositions completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error composing backgrounds: {e}")
            return False
    
    def compose_single_video(self, segment: Dict, script: Dict) -> bool:
        """
        Compose a single video with its background.
        
        Args:
            segment: Segment configuration dictionary
            script: Complete script configuration
            
        Returns:
            bool: Success status
        """
        try:
            segment_id = segment['id']
            background_file = segment['background']
            background_style = segment.get('background_style', 'full')
            speaker_position = segment.get('speaker_position', 'center')
            
            # Paths
            lipsync_path = Path(f"temp/lipsync/segment_{segment_id}.mp4")
            background_path = Path(f"input/backgrounds/{background_file}")
            output_path = Path(f"temp/composed/segment_{segment_id}.mp4")
            
            # Validate inputs
            if not lipsync_path.exists():
                self.logger.error(f"Lip-sync video not found: {lipsync_path}")
                return False
            
            if not background_path.exists():
                self.logger.error(f"Background image not found: {background_path}")
                return False
            
            # Load video and background
            lipsync_clip = VideoFileClip(str(lipsync_path))
            
            # Process based on background style
            if background_style == 'full':
                final_clip = self.compose_full_background(lipsync_clip, background_path, speaker_position)
            elif background_style == 'green_screen':
                final_clip = self.compose_green_screen(lipsync_clip, background_path, speaker_position)
            elif background_style == 'pip':
                final_clip = self.compose_picture_in_picture(lipsync_clip, background_path, speaker_position)
            else:
                # Default to full background
                final_clip = self.compose_full_background(lipsync_clip, background_path, speaker_position)
            
            # Apply video settings from script
            video_settings = script.get('video_settings', {})
            final_clip = self.apply_video_settings(final_clip, video_settings)
            
            # Write final video
            final_clip.write_videofile(
                str(output_path),
                codec=video_settings.get('codec', 'libx264'),
                bitrate=video_settings.get('bitrate', '5000k'),
                audio_codec='aac',
                temp_audiofile='temp/temp_audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            lipsync_clip.close()
            final_clip.close()
            
            self.logger.info(f"Composed background for segment {segment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error composing single video: {e}")
            return False
    
    def compose_full_background(self, video_clip: VideoFileClip, background_path: Path, 
                               speaker_position: str) -> CompositeVideoClip:
        """
        Compose video with full background replacement.
        
        Args:
            video_clip: Input video clip
            background_path: Path to background image
            speaker_position: Position of speaker ('center', 'left', 'right')
            
        Returns:
            Composed video clip
        """
        try:
            # Load and prepare background
            background_img = Image.open(background_path)
            target_size = self.config['processing']['default_resolution']
            
            # Resize background to target resolution
            background_img = background_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create background clip
            background_clip = ImageClip(np.array(background_img))
            background_clip = background_clip.set_duration(video_clip.duration)
            
            # Remove background from speaker video
            speaker_clip = self.remove_video_background(video_clip)
            
            # Position speaker
            speaker_clip = self.position_speaker(speaker_clip, speaker_position, target_size)
            
            # Composite
            final_clip = CompositeVideoClip([background_clip, speaker_clip])
            
            return final_clip
            
        except Exception as e:
            self.logger.error(f"Error in full background composition: {e}")
            # Fallback to simple overlay
            return self.simple_overlay(video_clip, background_path)
    
    def compose_green_screen(self, video_clip: VideoFileClip, background_path: Path,
                            speaker_position: str) -> CompositeVideoClip:
        """
        Compose video with green screen effect.
        
        Args:
            video_clip: Input video clip
            background_path: Path to background image
            speaker_position: Position of speaker
            
        Returns:
            Composed video clip
        """
        try:
            # Load background
            background_img = Image.open(background_path)
            target_size = self.config['processing']['default_resolution']
            background_img = background_img.resize(target_size, Image.Resampling.LANCZOS)
            
            background_clip = ImageClip(np.array(background_img))
            background_clip = background_clip.set_duration(video_clip.duration)
            
            # Apply green screen effect to speaker
            speaker_clip = self.apply_green_screen_effect(video_clip)
            speaker_clip = self.position_speaker(speaker_clip, speaker_position, target_size)
            
            # Composite
            final_clip = CompositeVideoClip([background_clip, speaker_clip])
            
            return final_clip
            
        except Exception as e:
            self.logger.error(f"Error in green screen composition: {e}")
            return self.simple_overlay(video_clip, background_path)
    
    def compose_picture_in_picture(self, video_clip: VideoFileClip, background_path: Path,
                                  speaker_position: str) -> CompositeVideoClip:
        """
        Compose video with picture-in-picture style.
        
        Args:
            video_clip: Input video clip
            background_path: Path to background image
            speaker_position: Position of speaker
            
        Returns:
            Composed video clip
        """
        try:
            # Load background (larger)
            background_img = Image.open(background_path)
            target_size = self.config['processing']['default_resolution']
            background_img = background_img.resize(target_size, Image.Resampling.LANCZOS)
            
            background_clip = ImageClip(np.array(background_img))
            background_clip = background_clip.set_duration(video_clip.duration)
            
            # Resize speaker to smaller size (PiP)
            pip_ratio = self.config['background_settings'].get('speaker_size_ratio', 0.3)
            pip_size = (int(target_size[0] * pip_ratio), int(target_size[1] * pip_ratio))
            
            speaker_clip = video_clip.resize(pip_size)
            speaker_clip = self.position_pip_speaker(speaker_clip, speaker_position, target_size, pip_size)
            
            # Add border/shadow to PiP
            speaker_clip = self.add_pip_effects(speaker_clip)
            
            # Composite
            final_clip = CompositeVideoClip([background_clip, speaker_clip])
            
            return final_clip
            
        except Exception as e:
            self.logger.error(f"Error in picture-in-picture composition: {e}")
            return self.simple_overlay(video_clip, background_path)
    
    def remove_video_background(self, video_clip: VideoFileClip) -> VideoFileClip:
        """
        Remove background from video using rembg.
        
        Args:
            video_clip: Input video clip
            
        Returns:
            Video clip with background removed
        """
        try:
            def remove_bg_frame(get_frame, t):
                frame = get_frame(t)
                # Convert frame to PIL Image
                pil_frame = Image.fromarray((frame * 255).astype(np.uint8))
                # Remove background
                removed_bg = remove(pil_frame)
                # Convert back to numpy array
                return np.array(removed_bg) / 255.0
            
            # Apply background removal to all frames
            bg_removed_clip = video_clip.fl(remove_bg_frame)
            return bg_removed_clip
            
        except Exception as e:
            self.logger.warning(f"Background removal failed, using original video: {e}")
            return video_clip
    
    def apply_green_screen_effect(self, video_clip: VideoFileClip) -> VideoFileClip:
        """
        Apply green screen effect to video.
        
        Args:
            video_clip: Input video clip
            
        Returns:
            Video clip with green screen effect applied
        """
        try:
            def green_screen_frame(get_frame, t):
                frame = get_frame(t)
                # Simple green screen - replace green areas with transparency
                # This is a basic implementation
                green_mask = (frame[:, :, 1] > 0.7) & (frame[:, :, 0] < 0.3) & (frame[:, :, 2] < 0.3)
                alpha = np.ones(frame.shape[:2])
                alpha[green_mask] = 0
                
                # Add alpha channel
                frame_with_alpha = np.dstack([frame, alpha])
                return frame_with_alpha
            
            return video_clip.fl(green_screen_frame)
            
        except Exception as e:
            self.logger.warning(f"Green screen effect failed: {e}")
            return video_clip
    
    def position_speaker(self, speaker_clip: VideoFileClip, position: str, 
                        target_size: Tuple[int, int]) -> VideoFileClip:
        """
        Position speaker in the frame.
        
        Args:
            speaker_clip: Speaker video clip
            position: Position string ('center', 'left', 'right', 'top', 'bottom')
            target_size: Target frame size (width, height)
            
        Returns:
            Positioned video clip
        """
        try:
            if position == 'center':
                return speaker_clip.set_position('center')
            elif position == 'left':
                return speaker_clip.set_position(('left', 'center'))
            elif position == 'right':
                return speaker_clip.set_position(('right', 'center'))
            elif position == 'top':
                return speaker_clip.set_position(('center', 'top'))
            elif position == 'bottom':
                return speaker_clip.set_position(('center', 'bottom'))
            else:
                return speaker_clip.set_position('center')
                
        except Exception as e:
            self.logger.warning(f"Speaker positioning failed: {e}")
            return speaker_clip.set_position('center')
    
    def position_pip_speaker(self, speaker_clip: VideoFileClip, position: str,
                           target_size: Tuple[int, int], pip_size: Tuple[int, int]) -> VideoFileClip:
        """
        Position speaker for picture-in-picture mode.
        
        Args:
            speaker_clip: Speaker video clip
            position: Position string
            target_size: Target frame size
            pip_size: PiP size
            
        Returns:
            Positioned video clip
        """
        try:
            margin = 20  # Margin from edges
            
            if position == 'bottom_right':
                x = target_size[0] - pip_size[0] - margin
                y = target_size[1] - pip_size[1] - margin
            elif position == 'bottom_left':
                x = margin
                y = target_size[1] - pip_size[1] - margin
            elif position == 'top_right':
                x = target_size[0] - pip_size[0] - margin
                y = margin
            elif position == 'top_left':
                x = margin
                y = margin
            else:  # Default to bottom_right
                x = target_size[0] - pip_size[0] - margin
                y = target_size[1] - pip_size[1] - margin
            
            return speaker_clip.set_position((x, y))
            
        except Exception as e:
            self.logger.warning(f"PiP positioning failed: {e}")
            return speaker_clip.set_position('center')
    
    def add_pip_effects(self, speaker_clip: VideoFileClip) -> VideoFileClip:
        """
        Add visual effects to picture-in-picture clip.
        
        Args:
            speaker_clip: Speaker video clip
            
        Returns:
            Enhanced video clip
        """
        try:
            # Add subtle border effect
            # This would require more complex video processing
            # For now, return as-is
            return speaker_clip
            
        except Exception as e:
            self.logger.warning(f"PiP effects failed: {e}")
            return speaker_clip
    
    def simple_overlay(self, video_clip: VideoFileClip, background_path: Path) -> CompositeVideoClip:
        """
        Simple overlay composition as fallback.
        
        Args:
            video_clip: Input video clip
            background_path: Path to background image
            
        Returns:
            Composed video clip
        """
        try:
            background_img = Image.open(background_path)
            target_size = self.config['processing']['default_resolution']
            background_img = background_img.resize(target_size, Image.Resampling.LANCZOS)
            
            background_clip = ImageClip(np.array(background_img))
            background_clip = background_clip.set_duration(video_clip.duration)
            
            # Simple center overlay
            speaker_clip = video_clip.set_position('center')
            
            return CompositeVideoClip([background_clip, speaker_clip])
            
        except Exception as e:
            self.logger.error(f"Simple overlay failed: {e}")
            return video_clip
    
    def apply_video_settings(self, clip: VideoFileClip, settings: Dict) -> VideoFileClip:
        """
        Apply video settings and effects.
        
        Args:
            clip: Input video clip
            settings: Video settings dictionary
            
        Returns:
            Processed video clip
        """
        try:
            # Apply quality settings
            if settings.get('quality') == 'high':
                # High quality settings already handled in write_videofile
                pass
            
            # Apply any video effects
            # Could add filters, color correction, etc.
            
            return clip
            
        except Exception as e:
            self.logger.warning(f"Video settings application failed: {e}")
            return clip
    
    def enhance_background(self, background_img: Image.Image) -> Image.Image:
        """
        Enhance background image quality.
        
        Args:
            background_img: Input background image
            
        Returns:
            Enhanced background image
        """
        try:
            # Apply subtle blur for professional look
            if self.config['background_settings'].get('blur_background', False):
                background_img = background_img.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Adjust opacity
            opacity = self.config['background_settings'].get('background_opacity', 1.0)
            if opacity < 1.0:
                enhancer = ImageEnhance.Brightness(background_img)
                background_img = enhancer.enhance(opacity)
            
            return background_img
            
        except Exception as e:
            self.logger.warning(f"Background enhancement failed: {e}")
            return background_img


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compose videos with backgrounds')
    parser.add_argument('--script', default='input/script.json',
                       help='Path to script JSON file')
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize composer
    composer = BackgroundComposer(args.config)
    
    # Compose videos
    success = composer.compose_all_videos(args.script)
    
    if success:
        print("✅ Background composition completed successfully!")
    else:
        print("❌ Background composition failed!")
        exit(1)


if __name__ == "__main__":
    main() 