#!/usr/bin/env python3
"""
Video Assembler for Multi-Speaker Video Generation
Combines all composed video segments into a final video with transitions.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
from moviepy.editor import (VideoFileClip, concatenate_videoclips, 
                           CompositeVideoClip, TextClip, ColorClip)
from tqdm import tqdm

class VideoAssembler:
    def __init__(self, config_path: str = "config.json"):
        """Initialize Video Assembler with configuration."""
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
    
    def assemble_final_video(self, script_path: str, output_filename: Optional[str] = None) -> bool:
        """
        Assemble all video segments into a final video.
        
        Args:
            script_path: Path to the script JSON file
            output_filename: Optional custom output filename
            
        Returns:
            bool: Success status
        """
        try:
            # Load script
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            segments = script['segments']
            transitions = script.get('transitions', {})
            video_settings = script.get('video_settings', {})
            
            self.logger.info(f"Assembling final video from {len(segments)} segments...")
            
            # Load all video clips
            clips = self.load_video_clips(segments)
            if not clips:
                self.logger.error("No video clips loaded")
                return False
            
            # Apply transitions
            processed_clips = self.apply_transitions(clips, transitions)
            
            # Add intro/outro if specified
            if script.get('intro'):
                processed_clips = self.add_intro(processed_clips, script['intro'])
            
            if script.get('outro'):
                processed_clips = self.add_outro(processed_clips, script['outro'])
            
            # Concatenate all clips
            final_video = concatenate_videoclips(processed_clips, method="compose")
            
            # Apply final video effects
            final_video = self.apply_final_effects(final_video, script)
            
            # Determine output path
            if output_filename:
                output_path = Path("output") / output_filename
            else:
                project_name = script.get('project_name', 'multi_speaker_video')
                safe_name = self.make_safe_filename(project_name)
                output_path = Path("output") / f"{safe_name}.mp4"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write final video
            self.write_final_video(final_video, output_path, video_settings)
            
            # Clean up
            for clip in clips:
                clip.close()
            final_video.close()
            
            # Generate video metadata
            self.generate_metadata(output_path, script)
            
            self.logger.info(f"Final video assembled successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error assembling final video: {e}")
            return False
    
    def load_video_clips(self, segments: List[Dict]) -> List[VideoFileClip]:
        """
        Load all composed video clips.
        
        Args:
            segments: List of segment configurations
            
        Returns:
            List of video clips
        """
        clips = []
        
        try:
            for segment in tqdm(segments, desc="Loading video clips"):
                segment_id = segment['id']
                video_path = Path(f"temp/composed/segment_{segment_id}.mp4")
                
                if not video_path.exists():
                    self.logger.error(f"Composed video not found: {video_path}")
                    continue
                
                clip = VideoFileClip(str(video_path))
                clips.append(clip)
                
                self.logger.info(f"Loaded clip for segment {segment_id}: {clip.duration:.2f}s")
            
            return clips
            
        except Exception as e:
            self.logger.error(f"Error loading video clips: {e}")
            return []
    
    def apply_transitions(self, clips: List[VideoFileClip], 
                         transition_config: Dict) -> List[VideoFileClip]:
        """
        Apply transitions between video clips.
        
        Args:
            clips: List of video clips
            transition_config: Transition configuration
            
        Returns:
            List of clips with transitions applied
        """
        try:
            if len(clips) <= 1:
                return clips
            
            transition_type = transition_config.get('type', 'crossfade')
            transition_duration = transition_config.get('duration', 1.0)
            
            processed_clips = []
            
            for i, clip in enumerate(clips):
                if i == 0:
                    # First clip - no incoming transition
                    processed_clips.append(clip)
                else:
                    # Apply transition based on type
                    if transition_type == 'crossfade':
                        clip_with_transition = clip.fadein(transition_duration)
                    elif transition_type == 'fade_in':
                        clip_with_transition = clip.fadein(transition_duration)
                    elif transition_type == 'slide':
                        clip_with_transition = self.apply_slide_transition(clip, transition_duration)
                    else:
                        # Default to fade in
                        clip_with_transition = clip.fadein(transition_duration)
                    
                    processed_clips.append(clip_with_transition)
                
                # Apply outgoing transition (except for last clip)
                if i < len(clips) - 1:
                    if transition_type == 'crossfade':
                        # Crossfade will be handled by the next clip's crossfadein
                        pass
                    elif transition_type == 'fade_out':
                        processed_clips[-1] = processed_clips[-1].fadeout(transition_duration)
            
            self.logger.info(f"Applied {transition_type} transitions with duration {transition_duration}s")
            return processed_clips
            
        except Exception as e:
            self.logger.error(f"Error applying transitions: {e}")
            return clips
    
    def apply_slide_transition(self, clip: VideoFileClip, duration: float) -> VideoFileClip:
        """
        Apply slide transition effect.
        
        Args:
            clip: Input video clip
            duration: Transition duration
            
        Returns:
            Clip with slide transition
        """
        try:
            # Simple slide-in from right effect
            w, h = clip.size
            
            def make_frame(get_frame, t):
                if t < duration:
                    # Slide in from right
                    progress = t / duration
                    frame = get_frame(t)
                    # Create sliding effect by shifting the frame
                    slide_frame = np.zeros_like(frame)
                    slide_width = int(w * progress)
                    slide_frame[:, :slide_width] = frame[:, w-slide_width:w]
                    return slide_frame
                else:
                    return get_frame(t)
            
            return clip.fl(make_frame)
            
        except Exception as e:
            self.logger.warning(f"Slide transition failed: {e}")
            return clip
    
    def add_intro(self, clips: List[VideoFileClip], intro_config: Dict) -> List[VideoFileClip]:
        """
        Add intro to the video.
        
        Args:
            clips: List of video clips
            intro_config: Intro configuration
            
        Returns:
            List of clips with intro added
        """
        try:
            intro_type = intro_config.get('type', 'text')
            
            if intro_type == 'text':
                intro_clip = self.create_text_intro(intro_config)
            elif intro_type == 'video':
                intro_clip = VideoFileClip(intro_config['file'])
            else:
                self.logger.warning(f"Unknown intro type: {intro_type}")
                return clips
            
            return [intro_clip] + clips
            
        except Exception as e:
            self.logger.error(f"Error adding intro: {e}")
            return clips
    
    def add_outro(self, clips: List[VideoFileClip], outro_config: Dict) -> List[VideoFileClip]:
        """
        Add outro to the video.
        
        Args:
            clips: List of video clips
            outro_config: Outro configuration
            
        Returns:
            List of clips with outro added
        """
        try:
            outro_type = outro_config.get('type', 'text')
            
            if outro_type == 'text':
                outro_clip = self.create_text_outro(outro_config)
            elif outro_type == 'video':
                outro_clip = VideoFileClip(outro_config['file'])
            else:
                self.logger.warning(f"Unknown outro type: {outro_type}")
                return clips
            
            return clips + [outro_clip]
            
        except Exception as e:
            self.logger.error(f"Error adding outro: {e}")
            return clips
    
    def create_text_intro(self, intro_config: Dict) -> CompositeVideoClip:
        """
        Create text-based intro clip.
        
        Args:
            intro_config: Intro configuration
            
        Returns:
            Intro video clip
        """
        try:
            text = intro_config.get('text', 'Multi-Speaker Video')
            duration = intro_config.get('duration', 3.0)
            
            # Get video dimensions
            resolution = self.config['processing']['default_resolution']
            
            # Create background
            background = ColorClip(size=resolution, color=(0, 0, 0), duration=duration)
            
            # Create text
            text_clip = TextClip(
                text,
                fontsize=50,
                color='white',
                font='Arial-Bold'
            ).set_position('center').set_duration(duration)
            
            # Add fade effects
            text_clip = text_clip.fadein(0.5).fadeout(0.5)
            
            return CompositeVideoClip([background, text_clip])
            
        except Exception as e:
            self.logger.error(f"Error creating text intro: {e}")
            # Return simple black clip as fallback
            resolution = self.config['processing']['default_resolution']
            return ColorClip(size=resolution, color=(0, 0, 0), duration=3.0)
    
    def create_text_outro(self, outro_config: Dict) -> CompositeVideoClip:
        """
        Create text-based outro clip.
        
        Args:
            outro_config: Outro configuration
            
        Returns:
            Outro video clip
        """
        try:
            text = outro_config.get('text', 'Thank you for watching!')
            duration = outro_config.get('duration', 3.0)
            
            # Get video dimensions
            resolution = self.config['processing']['default_resolution']
            
            # Create background
            background = ColorClip(size=resolution, color=(0, 0, 0), duration=duration)
            
            # Create text
            text_clip = TextClip(
                text,
                fontsize=40,
                color='white',
                font='Arial'
            ).set_position('center').set_duration(duration)
            
            # Add fade effects
            text_clip = text_clip.fadein(0.5).fadeout(0.5)
            
            return CompositeVideoClip([background, text_clip])
            
        except Exception as e:
            self.logger.error(f"Error creating text outro: {e}")
            # Return simple black clip as fallback
            resolution = self.config['processing']['default_resolution']
            return ColorClip(size=resolution, color=(0, 0, 0), duration=3.0)
    
    def apply_final_effects(self, video: VideoFileClip, script: Dict) -> VideoFileClip:
        """
        Apply final effects to the assembled video.
        
        Args:
            video: Input video
            script: Script configuration
            
        Returns:
            Video with effects applied
        """
        try:
            # Apply any final color correction, stabilization, etc.
            
            # Ensure consistent frame rate
            target_fps = script.get('fps', self.config['processing']['default_fps'])
            if video.fps != target_fps:
                video = video.set_fps(target_fps)
            
            # Apply final fade in/out if specified
            if script.get('final_fade', False):
                video = video.fadein(0.5).fadeout(0.5)
            
            return video
            
        except Exception as e:
            self.logger.warning(f"Final effects application failed: {e}")
            return video
    
    def write_final_video(self, video: VideoFileClip, output_path: Path, 
                         video_settings: Dict):
        """
        Write the final video to file.
        
        Args:
            video: Final assembled video
            output_path: Output file path
            video_settings: Video encoding settings
        """
        try:
            self.logger.info(f"Writing final video to {output_path}")
            self.logger.info(f"Video duration: {video.duration:.2f}s")
            self.logger.info(f"Video resolution: {video.size}")
            self.logger.info(f"Video FPS: {video.fps}")
            
            # Write video with optimal settings
            video.write_videofile(
                str(output_path),
                codec=video_settings.get('codec', 'libx264'),
                bitrate=video_settings.get('bitrate', '5000k'),
                audio_codec='aac',
                temp_audiofile='temp/temp_final_audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None,
                preset='medium',
                ffmpeg_params=['-crf', '23']
            )
            
            self.logger.info(f"Final video written successfully: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error writing final video: {e}")
            raise
    
    def generate_metadata(self, video_path: Path, script: Dict):
        """
        Generate metadata file for the video.
        
        Args:
            video_path: Path to the video file
            script: Script configuration
        """
        try:
            metadata = {
                'title': script.get('project_name', 'Multi-Speaker Video'),
                'duration': self.get_video_duration(video_path),
                'segments': len(script['segments']),
                'speakers': [segment['speaker'] for segment in script['segments']],
                'backgrounds': [segment['background'] for segment in script['segments']],
                'creation_date': self.get_current_timestamp(),
                'settings': {
                    'resolution': script.get('output_resolution'),
                    'fps': script.get('fps'),
                    'codec': script.get('video_settings', {}).get('codec')
                }
            }
            
            metadata_path = video_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Metadata generation failed: {e}")
    
    def get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def make_safe_filename(self, filename: str) -> str:
        """Make filename safe for filesystem."""
        import re
        # Remove invalid characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace spaces with underscores
        safe_name = safe_name.replace(' ', '_')
        return safe_name.lower()
    
    def cleanup_temp_files(self):
        """Clean up temporary files if configured."""
        try:
            if self.config['advanced'].get('cleanup_temp', True):
                temp_dir = Path('temp')
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    self.logger.info("Temporary files cleaned up")
                    
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Assemble final video')
    parser.add_argument('--script', default='input/script.json',
                       help='Path to script JSON file')
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--output', 
                       help='Output filename (optional)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up temporary files after assembly')
    
    args = parser.parse_args()
    
    # Initialize assembler
    assembler = VideoAssembler(args.config)
    
    # Assemble final video
    success = assembler.assemble_final_video(args.script, args.output)
    
    # Cleanup if requested
    if args.cleanup:
        assembler.cleanup_temp_files()
    
    if success:
        print("✅ Video assembly completed successfully!")
    else:
        print("❌ Video assembly failed!")
        exit(1)


if __name__ == "__main__":
    main() 