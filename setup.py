#!/usr/bin/env python3
"""
Setup script for Multi-Speaker Lip-Sync Video Generator
Downloads and configures all required models and dependencies.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import logging

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_command(cmd, cwd=None, check=True):
    """Run shell command with error handling."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(e.stderr)
        raise

def download_file(url, destination, description="file"):
    """Download file with progress."""
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading {description}...")
    
    try:
        urllib.request.urlretrieve(url, destination)
        logger.info(f"✅ Downloaded {description}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {description}: {e}")
        return False

def setup_wav2lip():
    """Setup Wav2Lip repository and models."""
    logger = logging.getLogger(__name__)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Clone Wav2Lip repository
    wav2lip_path = models_dir / "Wav2Lip"
    if not wav2lip_path.exists():
        logger.info("Cloning Wav2Lip repository...")
        run_command([
            "git", "clone", 
            "https://github.com/Rudrabha/Wav2Lip.git",
            str(wav2lip_path)
        ])
        logger.info("✅ Wav2Lip repository cloned")
    else:
        logger.info("✅ Wav2Lip repository already exists")
    
    # Create checkpoints directory
    checkpoints_dir = wav2lip_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Download Wav2Lip model
    wav2lip_model_path = checkpoints_dir / "wav2lip_gan.pth"
    if not wav2lip_model_path.exists() or wav2lip_model_path.stat().st_size < 1000000:  # Less than 1MB
        logger.info("Downloading Wav2Lip model (this may take a while)...")
        
        # Try multiple download methods
        download_methods = [
            {
                "name": "Direct download from Hugging Face",
                "url": "https://huggingface.co/datasets/rudrabha/wav2lip/resolve/main/wav2lip_gan.pth",
                "method": "direct"
            },
            {
                "name": "Google Drive via gdown",
                "url": "1IWJBJgrX3aPz3CkEBrC0XjzqJ_8V8h",
                "method": "gdown"
            }
        ]
        
        success = False
        for method in download_methods:
            logger.info(f"Trying {method['name']}...")
            
            if method["method"] == "direct":
                success = download_file(method["url"], wav2lip_model_path, "Wav2Lip model")
            elif method["method"] == "gdown":
                try:
                    # Install gdown if not available
                    run_command([sys.executable, "-m", "pip", "install", "gdown"], check=False)
                    run_command(["gdown", method["url"], "-O", str(wav2lip_model_path)], check=False)
                    if wav2lip_model_path.exists() and wav2lip_model_path.stat().st_size > 1000000:
                        success = True
                        logger.info("✅ Downloaded using gdown")
                    else:
                        logger.warning("❌ gdown download failed or file too small")
                except Exception as e:
                    logger.warning(f"❌ gdown method failed: {e}")
            
            if success:
                break
        
        if not success:
            logger.warning("❌ Could not download Wav2Lip model automatically")
            logger.info("Please download manually using one of these methods:")
            logger.info("")
            logger.info("Method 1: Download from Google Drive")
            logger.info("1. Visit: https://drive.google.com/file/d/1IWJBJgrX3aPz3CkEBrC0XjzqJ_8V8h/view?usp=sharing")
            logger.info("2. Download the file")
            logger.info("3. Rename it to 'wav2lip_gan.pth'")
            logger.info(f"4. Place it in: {wav2lip_model_path}")
            logger.info("")
            logger.info("Method 2: Use gdown command")
            logger.info(f"gdown 1IWJBJgrX3aPz3CkEBrC0XjzqJ_8V8h -O {wav2lip_model_path}")
            logger.info("")
            logger.info("Method 3: Download from Hugging Face")
            logger.info("1. Visit: https://huggingface.co/datasets/rudrabha/wav2lip/blob/main/wav2lip_gan.pth")
            logger.info("2. Click 'Download' button")
            logger.info(f"3. Place it in: {wav2lip_model_path}")
            logger.info("")
            logger.info("Expected file size: ~116MB")
    else:
        logger.info("✅ Wav2Lip model already exists")
    
    # Download face detection model
    face_detection_path = models_dir / "face_detection.pth"
    if not face_detection_path.exists():
        face_detection_url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        success = download_file(face_detection_url, face_detection_path, "Face detection model")
        if not success:
            logger.warning("❌ Could not download face detection model")
    else:
        logger.info("✅ Face detection model already exists")

def install_dependencies():
    """Install Python dependencies."""
    logger = logging.getLogger(__name__)
    logger.info("Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        if Path("requirements.txt").exists():
            run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            logger.error("❌ requirements.txt not found")
            return False
        
        logger.info("✅ Dependencies installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    logger = logging.getLogger(__name__)
    logger.info("Setting up directories...")
    
    directories = [
        "input/speakers",
        "input/backgrounds", 
        "processing",
        "temp/audio",
        "temp/lipsync",
        "temp/composed",
        "output",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")

def create_sample_files():
    """Create sample input files for testing."""
    logger = logging.getLogger(__name__)
    logger.info("Creating sample files...")
    
    # Create sample README files
    sample_readme = """# Sample Input Files

Place your files in these directories:

## Speakers (input/speakers/)
- Add photos or videos of the people who will be speaking
- Supported formats: .jpg, .jpeg, .png, .mp4, .avi, .mov
- Recommended: High-quality headshots or talking head videos

## Backgrounds (input/backgrounds/)  
- Add photos of people being talked about
- Supported formats: .jpg, .jpeg, .png
- Recommended: High-resolution photos

## Example Usage:
1. Add speaker1.jpg, speaker2.jpg to input/speakers/
2. Add person1.jpg, person2.jpg to input/backgrounds/
3. Edit input/script.json to match your files
4. Run: python main.py
"""
    
    (Path("input/speakers") / "README.md").write_text(sample_readme)
    (Path("input/backgrounds") / "README.md").write_text(sample_readme)
    
    logger.info("✅ Sample files created")

def check_system_requirements():
    """Check system requirements."""
    logger = logging.getLogger(__name__)
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("❌ Python 3.7+ required")
        return False
    
    # Check git
    try:
        run_command(["git", "--version"], check=True)
        logger.info("✅ Git available")
    except:
        logger.error("❌ Git not found - please install Git")
        return False
    
    # Check ffmpeg
    try:
        run_command(["ffmpeg", "-version"], check=True)
        logger.info("✅ FFmpeg available")
    except:
        logger.warning("⚠️  FFmpeg not found - install for better video processing")
        logger.info("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
    
    logger.info("✅ System requirements check completed")
    return True

def main():
    """Main setup function."""
    logger = setup_logging()
    
    print("🚀 Multi-Speaker Lip-Sync Video Generator Setup")
    print("=" * 50)
    
    try:
        # Step 1: Check system requirements
        if not check_system_requirements():
            print("❌ System requirements not met")
            sys.exit(1)
        
        # Step 2: Setup directories
        setup_directories()
        
        # Step 3: Install dependencies
        if not install_dependencies():
            print("❌ Dependency installation failed")
            sys.exit(1)
        
        # Step 4: Setup Wav2Lip
        setup_wav2lip()
        
        # Step 5: Create sample files
        create_sample_files()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Add speaker photos/videos to input/speakers/")
        print("2. Add background photos to input/backgrounds/")
        print("3. Edit input/script.json with your content")
        print("4. Run: python main.py")
        print("\n📚 For more information, see the README.md file")
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 