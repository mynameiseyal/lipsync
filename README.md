# Multi-Speaker Lip-Sync Video Generator

🎬 **Create professional videos with multiple famous people speaking your text with custom backgrounds!**

This project allows you to generate videos where different speakers (famous people) take turns speaking your script, with each speaker having a custom background image showing the person they're talking about.

## ✨ Features

- **Sequential Multi-Speaker Videos**: Multiple speakers, one after another
- **Custom Backgrounds**: Each speaker gets a unique background image
- **High-Quality Lip-Sync**: Uses Wav2Lip for realistic lip synchronization
- **Multiple Voice Models**: Various TTS voices for different speakers
- **Professional Transitions**: Smooth crossfades between speakers
- **Flexible Configuration**: Easy JSON-based setup
- **Background Composition**: Multiple composition styles (full, green screen, picture-in-picture)

## 🎯 Use Cases

- **Educational Content**: Multiple experts discussing a topic
- **Documentary Style**: Famous figures talking about historical events
- **Product Reviews**: Different celebrities endorsing products
- **Storytelling**: Multiple narrators for different chapters
- **Tribute Videos**: Various speakers honoring someone

## 🚀 Quick Start

### 1. Setup
```bash
# Clone/download this project
git clone <your-repo-url>
cd lipsync

# Run setup (installs dependencies and downloads models)
python setup.py
```

### 2. Add Your Content
```bash
# Add speaker photos/videos
cp your_speaker1.jpg input/speakers/
cp your_speaker2.jpg input/speakers/

# Add background images  
cp background_person1.jpg input/backgrounds/
cp background_person2.jpg input/backgrounds/
```

### 3. Configure Your Script
Edit `input/script.json`:
```json
{
  "project_name": "My Amazing Video",
  "segments": [
    {
      "id": 1,
      "speaker": "speaker1.jpg",
      "background": "background_person1.jpg",
      "text": "Your first speaker's text here...",
      "voice_model": "tts_models/en/ljspeech/tacotron2-DDC"
    },
    {
      "id": 2, 
      "speaker": "speaker2.jpg",
      "background": "background_person2.jpg", 
      "text": "Your second speaker's text here...",
      "voice_model": "tts_models/en/vctk/vits"
    }
  ]
}
```

### 4. Generate Video
```bash
python main.py
```

Your final video will be in the `output/` folder! 🎉

## 📁 Project Structure

```
lipsync/
├── input/
│   ├── speakers/          # Speaker photos/videos
│   ├── backgrounds/       # Background images
│   └── script.json       # Your video script
├── processing/
│   ├── tts_generator.py   # Text-to-speech generation
│   ├── lipsync_engine.py  # Wav2Lip processing
│   ├── background_composer.py # Background composition
│   └── video_assembler.py # Final video assembly
├── temp/                  # Temporary processing files
├── output/               # Final videos
├── models/              # AI models (auto-downloaded)
├── main.py              # Main pipeline
├── setup.py             # Setup script
├── config.json          # Configuration
└── requirements.txt     # Dependencies
```

## 🔧 Configuration Options

### Script Configuration (`input/script.json`)

```json
{
  "project_name": "My Video Project",
  "output_resolution": [1280, 720],
  "fps": 25,
  "segments": [
    {
      "id": 1,
      "speaker": "speaker1.jpg",
      "background": "person1.jpg",
      "text": "Text for this segment...",
      "voice_model": "tts_models/en/ljspeech/tacotron2-DDC",
      "speaker_position": "center",
      "background_style": "full"
    }
  ],
  "transitions": {
    "type": "crossfade",
    "duration": 1.0
  }
}
```

### Available Voice Models
- `tts_models/en/ljspeech/tacotron2-DDC` - Clear, neutral voice
- `tts_models/en/vctk/vits` - Varied speakers
- `tts_models/en/ljspeech/glow-tts` - Fast, good quality
- `tts_models/en/sam/tacotron-DDC` - Alternative voice

### Background Styles
- **`full`** - Full background replacement with speaker overlay
- **`green_screen`** - Green screen effect (removes speaker background)
- **`pip`** - Picture-in-picture style (small speaker, large background)

### Speaker Positions
- `center` - Speaker in center of frame
- `left`, `right` - Speaker on side
- `top`, `bottom` - Speaker at top/bottom

## 🎨 Advanced Usage

### Custom Output Name
```bash
python main.py --output "my_custom_video.mp4"
```

### Skip Steps (for debugging)
```bash
python main.py --skip tts,lipsync  # Skip TTS and lip-sync
```

### Check Progress
```bash
python main.py --progress
```

### Validate Inputs Only
```bash
python main.py --validate-only
```

## 🛠️ System Requirements

### Required
- **Python 3.7+**
- **Git** (for downloading models)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ disk space** (for models and temp files)

### Recommended
- **NVIDIA GPU** with CUDA support (faster processing)
- **FFmpeg** (better video processing)
- **16GB+ RAM** (for large videos)

### Supported File Formats

**Speakers:**
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Videos: `.mp4`, `.avi`, `.mov`

**Backgrounds:**
- Images: `.jpg`, `.jpeg`, `.png`

## 🔍 Troubleshooting

### Common Issues

**"Wav2Lip model not found"**
```bash
# Re-run setup to download models
python setup.py
```

**"CUDA out of memory"**
- Reduce video resolution in `config.json`
- Use smaller batch sizes
- Close other applications

**"Speaker file not found"**
- Check file paths in `script.json`
- Ensure files are in correct directories
- Check file extensions

**Poor lip-sync quality**
- Use high-quality speaker images/videos
- Ensure good lighting in speaker images
- Use clear, front-facing photos

### Debug Mode
Enable debug mode in `config.json`:
```json
{
  "advanced": {
    "debug_mode": true
  }
}
```

### Log Files
Check `pipeline.log` for detailed error information.

## 🎬 Tips for Best Results

### Speaker Images/Videos
- **High resolution** (1080p+)
- **Good lighting** 
- **Front-facing** shots
- **Clear facial features**
- **Minimal background** (for better processing)

### Background Images
- **High resolution** (1080p+)
- **Good contrast** with speaker
- **Relevant to content**

### Text Content
- **Natural speech patterns**
- **Appropriate length** (30-60 seconds per segment)
- **Clear pronunciation** words

## 🚀 Performance Optimization

### For Faster Processing
```json
{
  "processing": {
    "default_resolution": [854, 480]
  },
  "advanced": {
    "gpu_acceleration": true,
    "batch_processing": true
  }
}
```

### For Better Quality
```json
{
  "processing": {
    "default_resolution": [1920, 1080]
  },
  "video_settings": {
    "quality": "high",
    "bitrate": "10000k"
  }
}
```

## 📚 Examples

### Educational Video
```json
{
  "project_name": "History of AI",
  "segments": [
    {
      "id": 1,
      "speaker": "einstein.jpg",
      "background": "alan_turing.jpg",
      "text": "Today we explore the brilliant mind of Alan Turing, the father of computer science..."
    }
  ]
}
```

### Product Review
```json
{
  "project_name": "iPhone Review",
  "segments": [
    {
      "id": 1,
      "speaker": "tech_reviewer.jpg", 
      "background": "iphone.jpg",
      "text": "This new iPhone represents a breakthrough in mobile technology..."
    }
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and personal use. Please respect the rights of individuals whose likeness you use.

## ⚠️ Ethical Use

- Only use images/videos you have rights to
- Respect privacy and consent
- Don't create misleading or harmful content
- Follow local laws and regulations

## 🆘 Support

- **Issues**: Create a GitHub issue
- **Documentation**: Check this README
- **Logs**: Review `pipeline.log` for errors
- **Community**: Join our discussions

---

🎬 **Happy video creating!** 

Made with ❤️ for content creators and AI enthusiasts. 