#!/usr/bin/env python3
"""
Main Pipeline for Multi-Speaker Lip-Sync Video Generation
Orchestrates the complete process from text input to final video.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Import our processing modules
from processing.tts_generator import TTSGenerator
from processing.lipsync_engine import LipSyncEngine
from processing.background_composer import BackgroundComposer
from processing.video_assembler import VideoAssembler
from processing.caption_generator import CaptionGenerator
from processing.gpu_utils import setup_gpu_environment

class MultiSpeakerPipeline:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the multi-speaker video pipeline."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Initialize GPU environment
        self.gpu_manager = setup_gpu_environment(self.config)
        self.logger.info(f"[INIT] Device initialized: {self.gpu_manager.get_device_info()}")
        
        # Initialize all components with GPU manager
        self.tts_generator = TTSGenerator(config_path, self.gpu_manager)
        self.lipsync_engine = LipSyncEngine(config_path, self.gpu_manager)
        self.background_composer = BackgroundComposer(config_path, self.gpu_manager)
        self.caption_generator = CaptionGenerator(config_path, self.gpu_manager)
        self.video_assembler = VideoAssembler(config_path, self.gpu_manager)
        
        self.logger.info("Multi-Speaker Pipeline initialized")
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config['advanced'].get('debug_mode', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_inputs(self, script_path: str) -> bool:
        """
        Validate all required inputs before processing.
        
        Args:
            script_path: Path to script JSON file
            
        Returns:
            bool: Validation success
        """
        try:
            self.logger.info("Validating inputs...")
            
            # Check script file
            if not Path(script_path).exists():
                self.logger.error(f"Script file not found: {script_path}")
                return False
            
            # Load and validate script
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            segments = script.get('segments', [])
            if not segments:
                self.logger.error("No segments found in script")
                return False
            
            # Validate each segment
            for segment in segments:
                # Check required fields
                required_fields = ['id', 'speaker', 'background', 'text']
                for field in required_fields:
                    if field not in segment:
                        self.logger.error(f"Missing required field '{field}' in segment {segment.get('id', 'unknown')}")
                        return False
                
                # Check speaker file exists
                speaker_path = Path(f"input/speakers/{segment['speaker']}")
                if not speaker_path.exists():
                    self.logger.error(f"Speaker file not found: {speaker_path}")
                    return False
                
                # Check background file exists (unless it's "none")
                if segment['background'].lower() != 'none':
                    background_path = Path(f"input/backgrounds/{segment['background']}")
                    if not background_path.exists():
                        self.logger.error(f"Background file not found: {background_path}")
                        return False
                
                # Validate text content
                if not segment['text'].strip():
                    self.logger.error(f"Empty text in segment {segment['id']}")
                    return False
            
            self.logger.info(f"✅ All inputs validated successfully ({len(segments)} segments)")
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def create_video(self, script_path: str, output_filename: Optional[str] = None, 
                    skip_steps: Optional[list] = None) -> bool:
        """
        Execute the complete video creation pipeline.
        
        Args:
            script_path: Path to script JSON file
            output_filename: Optional custom output filename
            skip_steps: Optional list of steps to skip (for debugging)
            
        Returns:
            bool: Success status
        """
        try:
            start_time = time.time()
            skip_steps = skip_steps or []
            
            self.logger.info("[PIPELINE] Starting Multi-Speaker Video Creation Pipeline")
            self.logger.info(f"Script: {script_path}")
            
            # Step 0: Validate inputs
            if not self.validate_inputs(script_path):
                return False
            
            # Step 1: Generate TTS audio
            if 'tts' not in skip_steps:
                self.logger.info("🎤 Step 1: Generating TTS audio...")
                if not self.tts_generator.generate_audio_segments(script_path):
                    self.logger.error("❌ TTS generation failed")
                    return False
                self.logger.info("✅ TTS generation completed")
            else:
                self.logger.info("⏭️  Skipping TTS generation")
            
            # Step 2: Create lip-sync videos
            if 'lipsync' not in skip_steps:
                self.logger.info("👄 Step 2: Creating lip-sync videos...")
                if not self.lipsync_engine.create_lipsync_videos(script_path):
                    self.logger.error("❌ Lip-sync generation failed")
                    return False
                self.logger.info("✅ Lip-sync generation completed")
            else:
                self.logger.info("⏭️  Skipping lip-sync generation")
            
            # Step 3: Compose backgrounds
            if 'background' not in skip_steps:
                self.logger.info("🖼️  Step 3: Composing backgrounds...")
                if not self.background_composer.compose_all_videos(script_path):
                    self.logger.error("❌ Background composition failed")
                    return False
                self.logger.info("✅ Background composition completed")
            else:
                self.logger.info("⏭️  Skipping background composition")
            
            # Step 4: Add captions
            if 'captions' not in skip_steps:
                self.logger.info("📝 Step 4: Adding captions...")
                if not self.caption_generator.add_captions_to_videos(script_path):
                    self.logger.error("❌ Caption generation failed")
                    return False
                self.logger.info("✅ Caption generation completed")
            else:
                self.logger.info("⏭️  Skipping caption generation")
            
            # Step 5: Assemble final video
            if 'assembly' not in skip_steps:
                self.logger.info("🎞️  Step 5: Assembling final video...")
                if not self.video_assembler.assemble_final_video(script_path, output_filename):
                    self.logger.error("❌ Video assembly failed")
                    return False
                self.logger.info("✅ Video assembly completed")
            else:
                self.logger.info("⏭️  Skipping video assembly")
            
            # Pipeline completed
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info(f"🎉 Pipeline completed successfully in {total_time:.2f} seconds!")
            
            # Optional cleanup
            if self.config['advanced'].get('cleanup_temp', True):
                self.logger.info("🧹 Cleaning up temporary files...")
                self.video_assembler.cleanup_temp_files()
            
            # GPU cleanup
            self.gpu_manager.cleanup()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def get_progress_info(self, script_path: str) -> Dict:
        """
        Get progress information about the current pipeline state.
        
        Args:
            script_path: Path to script JSON file
            
        Returns:
            Dictionary with progress information
        """
        try:
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            segments = script['segments']
            total_segments = len(segments)
            
            # Check progress of each step
            progress = {
                'total_segments': total_segments,
                'audio_generated': 0,
                'lipsync_created': 0,
                'background_composed': 0,
                'final_video_exists': False
            }
            
            # Check audio files
            for segment in segments:
                audio_path = Path(f"temp/audio/segment_{segment['id']}.wav")
                if audio_path.exists():
                    progress['audio_generated'] += 1
            
            # Check lip-sync videos
            for segment in segments:
                lipsync_path = Path(f"temp/lipsync/segment_{segment['id']}.mp4")
                if lipsync_path.exists():
                    progress['lipsync_created'] += 1
            
            # Check composed videos
            for segment in segments:
                composed_path = Path(f"temp/composed/segment_{segment['id']}.mp4")
                if composed_path.exists():
                    progress['background_composed'] += 1
            
            # Check final video
            project_name = script.get('project_name', 'multi_speaker_video')
            safe_name = self.video_assembler.make_safe_filename(project_name)
            final_path = Path(f"output/{safe_name}.mp4")
            progress['final_video_exists'] = final_path.exists()
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Error getting progress info: {e}")
            return {}
    
    def print_progress_status(self, script_path: str):
        """Print current progress status."""
        progress = self.get_progress_info(script_path)
        
        if not progress:
            print("❌ Could not get progress information")
            return
        
        total = progress['total_segments']
        
        print("\n📊 Pipeline Progress Status:")
        print(f"   🎤 Audio Generation:     {progress['audio_generated']}/{total} segments")
        print(f"   👄 Lip-sync Creation:    {progress['lipsync_created']}/{total} segments")
        print(f"   🖼️  Background Composition: {progress['background_composed']}/{total} segments")
        print(f"   🎞️  Final Video:         {'✅ Created' if progress['final_video_exists'] else '❌ Not created'}")
        
        # Calculate overall progress
        steps_completed = (
            (progress['audio_generated'] / total) +
            (progress['lipsync_created'] / total) + 
            (progress['background_composed'] / total) +
            (1 if progress['final_video_exists'] else 0)
        )
        overall_progress = (steps_completed / 4) * 100
        
        print(f"\n   📈 Overall Progress: {overall_progress:.1f}%")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Multi-Speaker Lip-Sync Video Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use default script.json
  python main.py --script my_script.json  # Use custom script
  python main.py --output my_video.mp4    # Custom output name
  python main.py --skip tts,lipsync       # Skip TTS and lip-sync steps
  python main.py --progress               # Show progress only
  python main.py --gpu-info               # Show GPU information
  python main.py --validate-only          # Validate inputs only
        """
    )
    
    parser.add_argument('--script', default='input/script.json',
                       help='Path to script JSON file (default: input/script.json)')
    parser.add_argument('--config', default='config.json',
                       help='Path to configuration file (default: config.json)')
    parser.add_argument('--output',
                       help='Output video filename (optional)')
    parser.add_argument('--skip',
                       help='Comma-separated list of steps to skip (tts,lipsync,background,assembly)')
    parser.add_argument('--progress', action='store_true',
                       help='Show progress status and exit')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate inputs and exit')
    parser.add_argument('--gpu-info', action='store_true',
                       help='Show GPU information and exit')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = MultiSpeakerPipeline(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Handle GPU info check
    if args.gpu_info:
        gpu_info = pipeline.gpu_manager.get_device_info()
        print("\n[GPU] GPU Information:")
        print(f"   Device: {gpu_info['device']}")
        print(f"   Name: {gpu_info['device_name']}")
        print(f"   GPU Available: {'Yes' if gpu_info['is_gpu'] else 'No'}")
        print(f"   CUDA Available: {'Yes' if gpu_info['cuda_available'] else 'No'}")
        if gpu_info['is_gpu']:
            print(f"   Memory: {gpu_info['memory_available'] / 1e9:.1f} GB")
            allocated, cached = pipeline.gpu_manager.get_memory_usage()
            print(f"   Memory Used: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
        return
    
    # Handle progress check
    if args.progress:
        pipeline.print_progress_status(args.script)
        return
    
    # Handle validation only
    if args.validate_only:
        if pipeline.validate_inputs(args.script):
            print("[SUCCESS] All inputs are valid")
        else:
            print("[ERROR] Input validation failed")
            sys.exit(1)
        return
    
    # Parse skip steps
    skip_steps = []
    if args.skip:
        skip_steps = [step.strip() for step in args.skip.split(',')]
    
    # Run the pipeline
    print("[LIPSYNC] Starting Multi-Speaker Lip-Sync Video Generation...")
    print(f"[SCRIPT] Script: {args.script}")
    if args.output:
        print(f"[OUTPUT] Output: {args.output}")
    if skip_steps:
        print(f"[SKIP] Skipping: {', '.join(skip_steps)}")
    print()
    
    success = pipeline.create_video(args.script, args.output, skip_steps)
    
    if success:
        print("\n[SUCCESS] Video creation completed successfully!")
        print(f"[OUTPUT] Check the 'output' folder for your final video.")
    else:
        print("\n[ERROR] Video creation failed!")
        print("[INFO] Check the pipeline.log file for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main() 