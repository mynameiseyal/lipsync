#!/usr/bin/env python3
"""
Lip-Sync Engine for Multi-Speaker Video Generation
Uses Wav2Lip to create lip-synced videos from speaker images/videos and audio.
"""

import json
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import torch
from tqdm import tqdm
import shutil

class LipSyncEngine:
    def __init__(self, config_path: str = "config.json", gpu_manager=None):
        """Initialize Lip-Sync Engine with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.get_device() if gpu_manager else torch.device('cpu')
        self.wav2lip_ready = self.setup_wav2lip()
        
        if not self.wav2lip_ready:
            self.logger.warning("⚠️  Wav2Lip is not ready. Some features will be unavailable.")
            self.logger.info("Please follow the instructions above to download the required model.")
    
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
    
    def setup_wav2lip(self):
        """Setup Wav2Lip repository and models."""
        self.wav2lip_path = Path(self.config['paths']['wav2lip_repo'])
        self.checkpoint_path = Path(self.config['paths']['wav2lip_checkpoint'])
        
        # Check if Wav2Lip is available
        if not self.wav2lip_path.exists():
            self.logger.error("Wav2Lip repository not found. Please run setup script first:")
            self.logger.error("python setup.py")
            return False
        
        # Check if models are available
        if not self.checkpoint_path.exists():
            self.logger.error("Wav2Lip checkpoint not found!")
            self.logger.error("Please download the model from:")
            self.logger.error("https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
            self.logger.error(f"Save it to: {self.checkpoint_path}")
            # Create a placeholder file to avoid immediate crashes
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_text("# Placeholder - Download actual model from Wav2Lip repository")
            return False
        
        # Check if the checkpoint file is actually a model (not our placeholder)
        if self.checkpoint_path.stat().st_size < 1000:  # Model should be much larger
            self.logger.error("Wav2Lip checkpoint appears to be invalid (too small).")
            self.logger.error("Please download the actual model from:")
            self.logger.error("https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
            self.logger.error(f"Save it to: {self.checkpoint_path}")
            return False
        
        self.logger.info("✅ Wav2Lip setup completed successfully")
        return True
    
    def create_lipsync_videos(self, script_path: str) -> bool:
        """
        Create lip-synced videos for all segments in the script.
        
        Args:
            script_path: Path to the script JSON file
            
        Returns:
            bool: Success status
        """
        if not self.wav2lip_ready:
            self.logger.error("❌ Wav2Lip is not ready. Cannot create lip-sync videos.")
            self.logger.error("Please download the required model first:")
            self.logger.error("https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
            self.logger.error(f"Save it to: {self.config['paths']['wav2lip_checkpoint']}")
            return False
            
        try:
            # Load script
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            segments = script['segments']
            
            self.logger.info(f"Creating lip-sync videos for {len(segments)} segments...")
            
            # Create output directory
            lipsync_dir = Path("temp/lipsync")
            lipsync_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each segment
            for segment in tqdm(segments, desc="Generating lip-sync videos"):
                success = self.create_single_lipsync(segment)
                if not success:
                    self.logger.error(f"Failed to create lip-sync for segment {segment['id']}")
                    return False
            
            self.logger.info("All lip-sync videos created successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating lip-sync videos: {e}")
            return False
    
    def create_single_lipsync(self, segment: Dict) -> bool:
        """
        Create lip-sync video for a single segment.
        
        Args:
            segment: Segment configuration dictionary
            
        Returns:
            bool: Success status
        """
        try:
            segment_id = segment['id']
            speaker_file = segment['speaker']
            
            # Paths
            speaker_path = Path(f"input/speakers/{speaker_file}")
            audio_path = Path(f"temp/audio/segment_{segment_id}.wav")
            output_path = Path(f"temp/lipsync/segment_{segment_id}.mp4")
            
            # Validate inputs
            if not speaker_path.exists():
                self.logger.error(f"Speaker file not found: {speaker_path}")
                return False
            
            if not audio_path.exists():
                self.logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Prepare speaker video if it's an image
            processed_speaker_path = self.prepare_speaker_input(speaker_path, segment_id)
            
            # Run Wav2Lip
            success = self.run_wav2lip(processed_speaker_path, audio_path, output_path)
            
            if success:
                # Post-process the output
                self.post_process_lipsync(output_path, segment)
                self.logger.info(f"Created lip-sync video for segment {segment_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating single lip-sync: {e}")
            return False
    
    def prepare_speaker_input(self, speaker_path: Path, segment_id: int) -> Path:
        """
        Prepare speaker input (convert image to video if needed).
        
        Args:
            speaker_path: Path to speaker image/video
            segment_id: Segment ID
            
        Returns:
            Path to processed speaker file
        """
        file_extension = speaker_path.suffix.lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Convert image to video
            output_path = Path(f"temp/speaker_video_{segment_id}.mp4")
            return self.image_to_video(speaker_path, output_path)
        else:
            # Use video as-is
            return speaker_path
    
    def image_to_video(self, image_path: Path, output_path: Path, duration: float = 10.0) -> Path:
        """
        Convert image to video for Wav2Lip processing.
        
        Args:
            image_path: Path to input image
            output_path: Path to output video
            duration: Video duration in seconds
            
        Returns:
            Path to created video
        """
        try:
            # Load and resize image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get target resolution
            target_resolution = self.config['processing']['default_resolution']
            img_resized = cv2.resize(img, tuple(target_resolution))
            
            # Create video writer
            fps = self.config['processing']['default_fps']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, tuple(target_resolution))
            
            # Write frames
            total_frames = int(fps * duration)
            for _ in range(total_frames):
                out.write(img_resized)
            
            out.release()
            self.logger.info(f"Converted image to video: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error converting image to video: {e}")
            return image_path
    
    def run_wav2lip(self, speaker_path: Path, audio_path: Path, output_path: Path) -> bool:
        """
        Run Wav2Lip inference with GPU acceleration when available.
        """
        try:
            import sys
            # Build command with GPU/CPU settings
            cmd = [
                sys.executable, "inference.py",
                "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
                "--face", f"../../{speaker_path}",
                "--audio", f"../../{audio_path}",
                "--outfile", f"../../{output_path}"
            ]
            
            # Add batch size and quality settings
            if self.gpu_manager and self.gpu_manager.is_gpu_available:
                # Use GPU with optimal batch size
                batch_size = self.gpu_manager.get_optimal_batch_size()
                cmd.extend(["--wav2lip_batch_size", str(batch_size)])
                
                # Memory monitoring
                self.gpu_manager.monitor_memory("before Wav2Lip")
                self.logger.info(f"🚀 Running Wav2Lip on {self.gpu_manager.device_name} (batch_size={batch_size})")
            else:
                # Use CPU with smaller batch size
                cmd.extend(["--wav2lip_batch_size", "16"])
                self.logger.info("🔄 Running Wav2Lip on CPU")
            
            # Add quality settings from config
            if self.config['processing'].get('max_resolution'):
                # Use resize_factor to control output resolution
                cmd.extend(["--resize_factor", "1"])
            
            # Run command
            self.logger.info(f"Running Wav2Lip for {output_path.name}...")
            
            # Set environment for subprocess
            env = os.environ.copy()
            if self.gpu_manager and self.gpu_manager.is_gpu_available:
                env['CUDA_VISIBLE_DEVICES'] = str(self.device).split(':')[-1] if ':' in str(self.device) else '0'
            
            result = subprocess.run(
                cmd,
                cwd=str(self.wav2lip_path),
                capture_output=True,
                text=True,
                timeout=600,  # Increased timeout for GPU processing
                env=env
            )
            
            # Monitor GPU memory after processing
            if self.gpu_manager and self.gpu_manager.is_gpu_available:
                self.gpu_manager.monitor_memory("after Wav2Lip")
            
            if result.returncode == 0:
                self.logger.info(f"✅ Successfully created lip-sync video: {output_path}")
                # Log Wav2Lip output for debugging
                if result.stdout:
                    self.logger.info(f"Wav2Lip output: {result.stdout}")
                return True
            else:
                self.logger.error(f"❌ Wav2Lip failed: {result.stderr}")
                if result.stdout:
                    self.logger.error(f"Wav2Lip stdout: {result.stdout}")
                # If GPU failed, try CPU fallback
                if self.gpu_manager and self.gpu_manager.is_gpu_available and self.config['gpu_settings'].get('fallback_to_cpu', True):
                    self.logger.info("🔄 Retrying with CPU fallback...")
                    return self._run_wav2lip_cpu_fallback(speaker_path, audio_path, output_path)
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Wav2Lip process timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running Wav2Lip: {e}")
            return False
    
    def _run_wav2lip_cpu_fallback(self, speaker_path: Path, audio_path: Path, output_path: Path) -> bool:
        """
        Run Wav2Lip inference with CPU fallback.
        """
        try:
            import sys
            cmd = [
                sys.executable, "inference.py",
                "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
                "--face", f"../../{speaker_path}",
                "--audio", f"../../{audio_path}",
                "--outfile", f"../../{output_path}",
                "--wav2lip_batch_size", "8"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.wav2lip_path),
                capture_output=True,
                text=True,
                timeout=900  # Longer timeout for CPU
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Successfully created lip-sync video with CPU: {output_path}")
                return True
            else:
                self.logger.error(f"❌ CPU fallback also failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in CPU fallback: {e}")
            return False
    
    def post_process_lipsync(self, video_path: Path, segment: Dict):
        """
        Post-process lip-sync video.
        
        Args:
            video_path: Path to the video file
            segment: Segment configuration
        """
        try:
            # Apply any video filters or corrections
            self.enhance_video_quality(video_path)
            
            # Apply face padding if specified
            face_padding = self.config['processing'].get('face_padding', [0, 0, 0, 0])
            if any(face_padding):
                self.apply_face_padding(video_path, face_padding)
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed for {video_path}: {e}")
    
    def enhance_video_quality(self, video_path: Path):
        """
        Enhance video quality using ffmpeg.
        
        Args:
            video_path: Path to the video file
        """
        try:
            temp_path = video_path.with_suffix('.temp.mp4')
            
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-y", str(temp_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with enhanced version
                shutil.move(str(temp_path), str(video_path))
                self.logger.info(f"Enhanced video quality: {video_path.name}")
            else:
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            self.logger.warning(f"Video enhancement failed: {e}")
    
    def apply_face_padding(self, video_path: Path, padding: List[int]):
        """
        Apply padding around detected face.
        
        Args:
            video_path: Path to the video file
            padding: Padding values [top, right, bottom, left]
        """
        # This would implement face padding logic
        # For now, we'll skip this advanced feature
        pass
    
    def get_video_info(self, video_path: Path) -> Dict:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")
            return {}


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate lip-sync videos')
    parser.add_argument('--script', default='input/script.json',
                       help='Path to script JSON file')
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = LipSyncEngine(args.config)
    
    # Create lip-sync videos
    success = engine.create_lipsync_videos(args.script)
    
    if success:
        print("✅ Lip-sync video generation completed successfully!")
    else:
        print("❌ Lip-sync video generation failed!")
        exit(1)


if __name__ == "__main__":
    main() 