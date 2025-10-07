#!/bin/bash

echo "🎵 Multi-Dimensional Audio Analysis Agent - Installation Script"
echo "=============================================================="

# Check Python version
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1-2)
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv audio_env

# Activate virtual environment
source audio_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core audio processing libraries
echo "🎵 Installing core audio libraries..."
pip install librosa>=0.10.0 soundfile>=0.12.1 numpy>=1.24.0 scipy>=1.10.0 matplotlib>=3.7.0

# Install speech processing libraries
echo "🎤 Installing speech processing libraries..."
pip install speechrecognition>=3.10.0 pydub>=0.25.1 textblob>=0.17.1 nltk>=3.8

# Install additional scientific computing libraries
echo "🔬 Installing scientific computing libraries..."
pip install scikit-learn>=1.3.0 pandas>=2.0.0

# Install additional audio analysis libraries (optional, may require system dependencies)
echo "🔧 Installing additional audio libraries (optional)..."
pip install aubio || echo "⚠️  aubio installation failed (requires system dependencies)"
pip install madmom || echo "⚠️  madmom installation failed (optional)"

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️  NLTK data download failed: {e}')
"

# Check if ffmpeg is available for better audio format support
echo "🔊 Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg found - enhanced audio format support available"
else
    echo "⚠️  ffmpeg not found - limited audio format support"
    echo "   To install ffmpeg:"
    echo "   - Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   - macOS: brew install ffmpeg"
    echo "   - Windows: Download from https://ffmpeg.org/"
fi

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import librosa
import numpy as np
import soundfile as sf
try:
    import speech_recognition
    print('✅ All core libraries imported successfully')
except ImportError as e:
    print(f'⚠️  Import warning: {e}')
"

echo ""
echo "🎉 Installation Complete!"
echo "=========================================="
echo ""
echo "📋 Quick Start:"
echo "1. Activate environment: source audio_env/bin/activate"
echo "2. Run demo:            python audio_agent_lite.py"
echo "3. Run with your file:  python -c \"from audio_agent_lite import AudioAnalysisAgent; agent = AudioAnalysisAgent(); agent.analyze_audio('your_file.wav'); agent.print_summary()\""
echo ""
echo "📁 Files created:"
echo "   audio_agent_lite.py    - Main agent script"
echo "   requirements.txt       - Dependencies list"
echo "   README.md             - Documentation"
echo "   demo.py               - Demo script"
echo "   setup.py              - Package setup"
echo ""
echo "🆓 This agent uses only FREE and OPEN-SOURCE libraries!"
echo "   No API keys or paid services required."
echo ""

# Save activated environment state
echo "export AUDIO_AGENT_ENV=activated" > audio_env.status

echo "✨ Ready to analyze audio! 🎵"