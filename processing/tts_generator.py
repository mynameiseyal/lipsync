#!/usr/bin/env python3
"""
Enhanced TTS Generator with support for multiple TTS services
Supports Coqui TTS, ElevenLabs, Azure, Google Cloud, and Amazon Polly
with comprehensive male and female voice options and enhanced quality.
"""

import json
import os
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from gtts import gTTS
from pydub import AudioSegment
from tqdm import tqdm
import pyttsx3

class TTSGenerator:
    def __init__(self, config_path: str = "config.json", gpu_manager=None):
        """Initialize TTS Generator with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.gpu_manager = gpu_manager
        
        # Voice provider configurations
        self.voice_providers = self.setup_voice_providers()
        
        # Note: gTTS doesn't use GPU, but neural TTS models could benefit from GPU
        if gpu_manager and gpu_manager.is_gpu_available:
            self.logger.info("[GPU] GPU available for neural TTS models")
        else:
            self.logger.info("[CPU] Using CPU for TTS generation")
    
    def setup_voice_providers(self) -> Dict:
        """Setup available voice providers and their configurations."""
        return {
            "enhanced_system": {
                "enabled": True,
                "quality_modes": ["enhanced", "slow_clear", "ffmpeg_enhanced"],
                "voices": {
                    "male": {
                        "Fred": "com.apple.speech.synthesis.voice.Fred",
                        "Thomas": "com.apple.voice.compact.fr-FR.Thomas", 
                        "Daniel": "com.apple.voice.compact.en-GB.Daniel"
                    },
                    "female": {
                        "Samantha": "com.apple.voice.compact.en-US.Samantha",
                        "Karen": "com.apple.voice.compact.en-AU.Karen"
                    }
                }
            },
            "coqui": {
                "enabled": True,
                "models": {
                    "male": [
                        "tts_models/en/vctk/vits",
                        "tts_models/en/sam/tacotron-DDC",
                        "tts_models/en/ljspeech/tacotron2-DDC"
                    ],
                    "female": [
                        "tts_models/en/ljspeech/vits",
                        "tts_models/en/ljspeech/glow-tts",
                        "tts_models/en/vctk/vits"
                    ]
                }
            },
            "elevenlabs": {
                "enabled": False,  # Requires API key
                "api_key": os.getenv("ELEVENLABS_API_KEY"),
                "voices": {
                    "male": ["Adam", "Antoni", "Arnold", "Clyde", "Dom", "Josh", "Sam"],
                    "female": ["Bella", "Charlotte", "Dorothea", "Emily", "Grace", "Rachel", "Sarah"]
                }
            },
            "azure": {
                "enabled": False,  # Requires API key
                "api_key": os.getenv("AZURE_SPEECH_KEY"),
                "region": os.getenv("AZURE_SPEECH_REGION"),
                "voices": {
                    "male": ["en-US-GuyNeural", "en-US-DavisNeural", "en-GB-RyanNeural"],
                    "female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural"]
                }
            },
            "google": {
                "enabled": False,  # Requires API key
                "api_key": os.getenv("GOOGLE_CLOUD_API_KEY"),
                "voices": {
                    "male": ["en-US-Standard-B", "en-US-Wavenet-B", "en-US-Wavenet-D"],
                    "female": ["en-US-Standard-A", "en-US-Wavenet-A", "en-US-Wavenet-C"]
                }
            },
            "amazon": {
                "enabled": False,  # Requires AWS credentials
                "voices": {
                    "male": ["Matthew", "Justin", "Kevin", "Joey"],
                    "female": ["Joanna", "Salli", "Kendra", "Kimberly"]
                }
            },
            "gtts": {
                "enabled": True,
                "fallback": True
            }
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
    
    def get_available_voices(self, gender: str = None) -> Dict:
        """Get all available voices, optionally filtered by gender."""
        available = {}
        
        for provider, config in self.voice_providers.items():
            if not config.get("enabled", False):
                continue
                
            if provider == "coqui":
                if gender:
                    voices = {gender: config["models"].get(gender, [])}
                else:
                    voices = config["models"]
                available[provider] = {
                    "type": "local",
                    "voices": voices
                }
            elif provider in ["elevenlabs", "azure", "google", "amazon"]:
                if config.get("api_key") or provider == "amazon":
                    if gender:
                        voices = {gender: config["voices"].get(gender, [])}
                    else:
                        voices = config["voices"]
                    available[provider] = {
                        "type": "cloud",
                        "voices": voices
                    }
            elif provider == "gtts":
                available[provider] = {
                    "type": "fallback",
                    "voices": ["default"]
                }
        
        return available
    
    def generate_audio_segments(self, script_path: str) -> bool:
        """
        Generate audio files for all segments in the script using selected TTS service.
        
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
            
            self.logger.info(f"Generating audio for {len(segments)} segments...")
            
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
        Generate audio for a single segment using selected TTS service.
        
        Args:
            segment: Segment configuration dictionary
            audio_settings: Audio processing settings
            
        Returns:
            bool: Success status
        """
        try:
            segment_id = segment['id']
            text = segment['text']
            
            # Get voice preferences
            voice_provider = audio_settings.get('voice_provider', 'enhanced_system')
            voice_gender = audio_settings.get('voice_gender', 'female')
            voice_name = audio_settings.get('voice_name', None)
            quality_mode = audio_settings.get('quality_mode', 'enhanced')
            
            self.logger.info(f"Generating audio for segment {segment_id} using {voice_provider}...")
            
            # Try the preferred provider first
            if self.voice_providers.get(voice_provider, {}).get('enabled', False):
                if voice_provider == 'enhanced_system':
                    return self._generate_enhanced_system(segment_id, text, audio_settings, voice_gender, voice_name, quality_mode)
                elif voice_provider == 'coqui':
                    return self._generate_coqui(segment_id, text, audio_settings, voice_gender, voice_name)
                elif voice_provider == 'elevenlabs':
                    return self._generate_elevenlabs(segment_id, text, audio_settings, voice_gender, voice_name)
                elif voice_provider == 'azure':
                    return self._generate_azure(segment_id, text, audio_settings, voice_gender, voice_name)
                elif voice_provider == 'google':
                    return self._generate_google(segment_id, text, audio_settings, voice_gender, voice_name)
                elif voice_provider == 'amazon':
                    return self._generate_amazon(segment_id, text, audio_settings, voice_gender, voice_name)
            
            # Fallback to gTTS
            self.logger.warning(f"Provider {voice_provider} not available, falling back to gTTS")
            return self._generate_gtts(segment_id, text, audio_settings)
            
        except Exception as e:
            self.logger.error(f"Error generating segment audio: {e}")
            return False
    
    def _generate_coqui(self, segment_id: int, text: str, audio_settings: Dict, 
                       gender: str, voice_name: str = None) -> bool:
        """Generate audio using Coqui TTS with gender-specific models."""
        try:
            from TTS.api import TTS
            
            # Select model based on gender and preference
            if voice_name and voice_name in self.voice_providers['coqui']['models'].get(gender, []):
                model_name = voice_name
            else:
                # Use first available model for gender
                models = self.voice_providers['coqui']['models'].get(gender, [])
                if not models:
                    self.logger.warning(f"No {gender} voice models available, using default")
                    models = self.voice_providers['coqui']['models'].get('female', ['tts_models/en/ljspeech/vits'])
                model_name = models[0]
            
            # Initialize TTS
            tts = TTS(model_name=model_name, progress_bar=False)
            
            output_path = f"temp/audio/segment_{segment_id}.wav"
            
            # Generate speech
            tts.tts_to_file(text=text, file_path=output_path)
            
            # Apply post-processing
            audio = AudioSegment.from_wav(output_path)
            
            if audio_settings.get('normalize', True):
                audio = audio.normalize()
            
            sample_rate = audio_settings.get('sample_rate', 22050)
            audio = audio.set_frame_rate(sample_rate)
            
            audio.export(output_path, format="wav")
            
            self.logger.info(f"Generated Coqui TTS {gender} voice for segment {segment_id}: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating Coqui TTS audio: {e}")
            self.logger.warning("Falling back to gTTS")
            return self._generate_gtts(segment_id, text, audio_settings)
    
    def _generate_elevenlabs(self, segment_id: int, text: str, audio_settings: Dict,
                           gender: str, voice_name: str = None) -> bool:
        """Generate audio using ElevenLabs API."""
        try:
            api_key = self.voice_providers['elevenlabs']['api_key']
            if not api_key:
                raise ValueError("ElevenLabs API key not found")
            
            # Select voice
            voices = self.voice_providers['elevenlabs']['voices'].get(gender, [])
            if voice_name and voice_name in voices:
                voice_id = voice_name
            else:
                voice_id = voices[0] if voices else "Adam"
            
            # API endpoint
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # Save audio
            output_path = f"temp/audio/segment_{segment_id}.wav"
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # Convert to WAV if needed
            audio = AudioSegment.from_mp3(output_path)
            if audio_settings.get('normalize', True):
                audio = audio.normalize()
            
            sample_rate = audio_settings.get('sample_rate', 22050)
            audio = audio.set_frame_rate(sample_rate)
            audio.export(output_path, format="wav")
            
            self.logger.info(f"Generated ElevenLabs {gender} voice for segment {segment_id}: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating ElevenLabs audio: {e}")
            return self._generate_gtts(segment_id, text, audio_settings)
    
    def _generate_enhanced_system(self, segment_id: int, text: str, audio_settings: Dict,
                                gender: str, voice_name: str = None, quality_mode: str = "enhanced") -> bool:
        """Generate enhanced quality audio using system voices."""
        try:
            # Get voice configuration
            voices = self.voice_providers['enhanced_system']['voices'].get(gender, {})
            
            if voice_name and voice_name in voices:
                voice_id = voices[voice_name]
                selected_voice = voice_name
            elif voices:
                # Default to first available voice
                selected_voice = list(voices.keys())[0]
                voice_id = voices[selected_voice]
            else:
                raise ValueError(f"No {gender} voices available in enhanced system")
            
            self.logger.info(f"Generating enhanced {gender} voice: {selected_voice} (quality: {quality_mode})")
            
            # Quality settings
            quality_settings = {
                'enhanced': {'rate': 200, 'volume': 1.0, 'sample_rate': 44100},
                'slow_clear': {'rate': 160, 'volume': 1.0, 'sample_rate': 44100},
                'basic': {'rate': 180, 'volume': 0.9, 'sample_rate': 22050}
            }
            
            settings = quality_settings.get(quality_mode, quality_settings['enhanced'])
            
            # Generate with pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('voice', voice_id)
            engine.setProperty('rate', settings['rate'])
            engine.setProperty('volume', settings['volume'])
            
            # Create temp output file
            temp_output = f"temp/audio/segment_{segment_id}_temp.wav"
            engine.save_to_file(text, temp_output)
            engine.runAndWait()
            
            if not os.path.exists(temp_output):
                raise Exception("Failed to generate audio with pyttsx3")
            
            final_output = f"temp/audio/segment_{segment_id}.wav"
            
            # Apply FFmpeg enhancement if requested
            if quality_mode == "ffmpeg_enhanced":
                self._enhance_audio_with_ffmpeg(temp_output, final_output)
                os.remove(temp_output)
            else:
                os.rename(temp_output, final_output)
            
            # Apply audio settings
            if audio_settings.get('normalize', True) or settings['sample_rate'] != 22050:
                audio = AudioSegment.from_wav(final_output)
                
                if audio_settings.get('normalize', True):
                    audio = audio.normalize()
                
                target_sample_rate = audio_settings.get('sample_rate', settings['sample_rate'])
                if audio.frame_rate != target_sample_rate:
                    audio = audio.set_frame_rate(target_sample_rate)
                
                audio.export(final_output, format="wav")
            
            self.logger.info(f"Generated enhanced {selected_voice} voice for segment {segment_id}: {final_output}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced system audio: {e}")
            return self._generate_gtts(segment_id, text, audio_settings)
    
    def _enhance_audio_with_ffmpeg(self, input_file: str, output_file: str) -> bool:
        """Enhance audio quality using FFmpeg filters."""
        try:
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', input_file,  # Input file
                '-af', 'highpass=f=80,lowpass=f=8000,volume=1.2,dynaudnorm',  # Audio filters
                '-ar', '44100',  # Sample rate
                '-ac', '1',      # Mono
                '-b:a', '256k',  # Bitrate
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                self.logger.info(f"Enhanced audio with FFmpeg: {output_file}")
                return True
            else:
                self.logger.warning(f"FFmpeg enhancement failed, copying original")
                os.rename(input_file, output_file)
                return False
                
        except Exception as e:
            self.logger.warning(f"FFmpeg error, copying original: {e}")
            if os.path.exists(input_file):
                os.rename(input_file, output_file)
            return False
    
    def _generate_azure(self, segment_id: int, text: str, audio_settings: Dict,
                       gender: str, voice_name: str = None) -> bool:
        """Generate audio using Azure Cognitive Services."""
        try:
            api_key = self.voice_providers['azure']['api_key']
            region = self.voice_providers['azure']['region']
            
            if not api_key or not region:
                raise ValueError("Azure Speech API key or region not found")
            
            # Select voice
            voices = self.voice_providers['azure']['voices'].get(gender, [])
            if voice_name and voice_name in voices:
                voice = voice_name
            else:
                voice = voices[0] if voices else "en-US-JennyNeural"
            
            # This would require the azure-cognitiveservices-speech package
            # For now, return False to fall back to gTTS
            self.logger.warning("Azure TTS not fully implemented, falling back to gTTS")
            return self._generate_gtts(segment_id, text, audio_settings)
            
        except Exception as e:
            self.logger.error(f"Error generating Azure audio: {e}")
            return self._generate_gtts(segment_id, text, audio_settings)
    
    def _generate_google(self, segment_id: int, text: str, audio_settings: Dict,
                        gender: str, voice_name: str = None) -> bool:
        """Generate audio using Google Cloud Text-to-Speech."""
        try:
            api_key = self.voice_providers['google']['api_key']
            if not api_key:
                raise ValueError("Google Cloud API key not found")
            
            # This would require the google-cloud-texttospeech package
            # For now, return False to fall back to gTTS
            self.logger.warning("Google Cloud TTS not fully implemented, falling back to gTTS")
            return self._generate_gtts(segment_id, text, audio_settings)
            
        except Exception as e:
            self.logger.error(f"Error generating Google Cloud audio: {e}")
            return self._generate_gtts(segment_id, text, audio_settings)
    
    def _generate_amazon(self, segment_id: int, text: str, audio_settings: Dict,
                        gender: str, voice_name: str = None) -> bool:
        """Generate audio using Amazon Polly."""
        try:
            # This would require boto3 and AWS credentials
            # For now, return False to fall back to gTTS
            self.logger.warning("Amazon Polly not fully implemented, falling back to gTTS")
            return self._generate_gtts(segment_id, text, audio_settings)
            
        except Exception as e:
            self.logger.error(f"Error generating Amazon Polly audio: {e}")
            return self._generate_gtts(segment_id, text, audio_settings)
    
    def _generate_gtts(self, segment_id: int, text: str, audio_settings: Dict) -> bool:
        """Generate audio using gTTS (fallback)."""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            temp_path = f"temp/audio/segment_{segment_id}_temp.mp3"
            output_path = f"temp/audio/segment_{segment_id}.wav"
            
            tts.save(temp_path)
            
            # Convert to WAV format
            audio = AudioSegment.from_mp3(temp_path)
            
            # Apply basic processing
            if audio_settings.get('normalize', True):
                audio = audio.normalize()
            
            # Ensure consistent sample rate
            sample_rate = audio_settings.get('sample_rate', 22050)
            audio = audio.set_frame_rate(sample_rate)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            # Clean up temp file
            os.remove(temp_path)
            
            self.logger.info(f"Generated gTTS audio for segment {segment_id}: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating gTTS audio: {e}")
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
    parser.add_argument('--list-voices', action='store_true',
                       help='List all available voices')
    parser.add_argument('--gender', choices=['male', 'female'],
                       help='Filter voices by gender')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TTSGenerator(args.config)
    
    if args.list_voices:
        # List available voices
        voices = generator.get_available_voices(args.gender)
        print("\n🎤 Available Voice Providers:")
        print("=" * 50)
        
        for provider, config in voices.items():
            print(f"\n📡 {provider.upper()} ({config['type']}):")
            if config['type'] == 'local':
                for gender, models in config['voices'].items():
                    print(f"  {gender.capitalize()}: {', '.join(models)}")
            elif config['type'] == 'cloud':
                for gender, voice_list in config['voices'].items():
                    print(f"  {gender.capitalize()}: {', '.join(voice_list)}")
            elif config['type'] == 'fallback':
                print(f"  Default: {', '.join(config['voices'])}")
        
        print(f"\n💡 To use cloud providers, set environment variables:")
        print("   ELEVENLABS_API_KEY=your_key")
        print("   AZURE_SPEECH_KEY=your_key")
        print("   AZURE_SPEECH_REGION=your_region")
        print("   GOOGLE_CLOUD_API_KEY=your_key")
        return
    
    # Generate audio segments
    success = generator.generate_audio_segments(args.script)
    
    if success:
        print("✅ Audio generation completed successfully!")
    else:
        print("❌ Audio generation failed!")
        exit(1)


if __name__ == "__main__":
    main() 