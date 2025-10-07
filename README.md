# Multi-Dimensional Audio Analysis Agent

A comprehensive **completely free** audio analysis system that implements parallel deconstruction across three distinct dimensions simultaneously:

1. **Musical & Structural Analysis**
2. **Lyrical & Vocal Analysis** 
3. **Sonic & Spectral Analysis**

## 🎯 Features

### 📊 Musical & Structural Analysis
- **Rhythm & Groove Agent**: BPM detection, tempo mapping, onset detection
- **Harmony & Key Agent**: Key detection, chord analysis, tonal clarity
- **Structure Agent**: Song segmentation, MFCC-based structural analysis

### 🎤 Lyrical & Vocal Analysis
- **Voice Activity Detection**: Spectral analysis to identify vocal content
- **Speech Recognition**: Free transcription using Google Speech API (no API key required)
- **Text Analysis**: Sentiment analysis, word counting, language detection
- **Vocal Characteristics**: Pitch range analysis, vocal frequency detection

### 🔊 Sonic & Spectral Analysis
- **Spectral Features**: Centroid, rolloff, bandwidth, zero-crossing rate
- **Frequency Analysis**: FFT-based dominant frequency detection
- **Dynamic Range**: RMS energy analysis, loudness variability
- **Harmonic-Percussive Separation**: Energy ratio analysis

## 🆓 Completely Free

✅ **No API keys required**  
✅ **No paid services needed**  
✅ **All open-source libraries**  
✅ **Works offline**  
✅ **No usage limits**  

## 🚀 Quick Start

### One-Command Installation
```bash
# Clone or download the files, then:
chmod +x install.sh
./install.sh
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv audio_env
source audio_env/bin/activate

# Install core libraries
pip install librosa soundfile numpy scipy matplotlib
pip install speechrecognition pydub textblob nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### Basic Usage
```bash
# Activate environment
source audio_env/bin/activate

# Run demo with built-in test audio
python audio_agent_lite.py

# Analyze your own audio file
python example_usage.py
```

## 📋 Usage Examples

### Simple Analysis
```python
from audio_agent_lite import AudioAnalysisAgent

# Initialize agent
agent = AudioAnalysisAgent()

# Analyze audio file
results = agent.analyze_audio('your_audio.wav')

# Display summary
agent.print_summary()

# Save detailed results
agent.save_results('analysis_results.json')
```

### Advanced Usage
```python
# Initialize with custom sample rate
agent = AudioAnalysisAgent(sample_rate=44100)

# Analyze and access specific results
results = agent.analyze_audio('song.mp3')

# Access specific analysis dimensions
musical = results['musical_structural']
vocal = results['lyrical_vocal'] 
spectral = results['sonic_spectral']

# Extract specific metrics
tempo = musical['rhythm']['estimated_tempo']
key = musical['harmony']['estimated_key']
voice_ratio = vocal['voice_activity']['voice_presence_ratio']
spectral_centroid = spectral['spectral_features']['spectral_centroid']['mean']
```

## 📊 Sample Output

```
🎵 AUDIO ANALYSIS SUMMARY
============================================================
📁 File: sample_song.wav
⏱️  Duration: 30.45 seconds
🎚️  Sample Rate: 22050 Hz

🎼 MUSICAL & STRUCTURAL ANALYSIS
----------------------------------------
   Tempo: 120.5 BPM
   Onsets: 58
   Key: C
   Tonal Clarity: 2.85

🎤 LYRICAL & VOCAL ANALYSIS
----------------------------------------
   Voice Presence: 65.3%
   Contains Vocals: Yes
   Transcribed: "Hello world this is a test of the system..."
   Sentiment: Positive (0.34)

🔊 SONIC & SPECTRAL ANALYSIS
----------------------------------------
   Spectral Centroid: 2847 Hz
   Dominant Frequency: 440 Hz
   Dynamic Range: 18.7 dB
   Harmonic: 78.2% | Percussive: 21.8%
```

## 🎵 Supported Audio Formats

- **WAV** (native support)
- **MP3** (requires ffmpeg)
- **FLAC** (requires ffmpeg)
- **OGG** (requires ffmpeg)
- **M4A** (requires ffmpeg)

### Installing ffmpeg for additional format support:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## 🔧 Library Dependencies

### Core Audio Processing
- **librosa** - Audio analysis and feature extraction
- **soundfile** - Audio file I/O
- **numpy** - Numerical computing
- **scipy** - Scientific computing

### Speech & Text Processing
- **speechrecognition** - Free speech-to-text
- **textblob** - Text analysis and sentiment
- **nltk** - Natural language processing
- **pydub** - Audio format conversion

### Scientific Computing
- **matplotlib** - Visualization (optional)
- **scikit-learn** - Machine learning utilities
- **pandas** - Data manipulation (optional)

## 🎪 Demo Scripts

### Basic Demo
```bash
python audio_agent_lite.py
```
Runs analysis on built-in test audio with full 3-dimensional analysis.

### Comprehensive Examples
```bash
python example_usage.py
```
Creates and analyzes different types of audio:
- Harmonic music (chord progressions)
- Speech-like audio (formant patterns)  
- Percussion patterns (rhythmic content)

### Custom Analysis
```python
# Analyze specific aspects
agent = AudioAnalysisAgent()
results = agent.analyze_audio('your_file.wav')

# Musical analysis only
musical_data = results['musical_structural']
print(f"Detected key: {musical_data['harmony']['estimated_key']}")
print(f"Tempo: {musical_data['rhythm']['estimated_tempo']} BPM")

# Vocal analysis only  
vocal_data = results['lyrical_vocal']
if vocal_data['voice_activity']['likely_contains_vocals']:
    print("Contains vocals!")
    if 'transcribed_text' in vocal_data.get('speech_recognition', {}):
        print(f"Lyrics: {vocal_data['speech_recognition']['transcribed_text']}")

# Spectral analysis only
spectral_data = results['sonic_spectral']
print(f"Dominant frequency: {spectral_data['frequency_analysis']['dominant_frequency']} Hz")
```

## 🏗️ Architecture

The agent implements **Layer 1: Parallel Deconstruction** where three analysis engines run simultaneously:

```
Audio Input
     │
     ├── Musical & Structural Engine
     │   ├── Rhythm & Groove Agent
     │   ├── Harmony & Key Agent  
     │   └── Structure Agent
     │
     ├── Lyrical & Vocal Engine
     │   ├── Voice Activity Detection
     │   ├── Speech Recognition
     │   └── Text Analysis
     │
     └── Sonic & Spectral Engine
         ├── Spectral Features
         ├── Frequency Analysis
         └── Dynamic Analysis
```

## 📈 Performance

- **Real-time capable** for short audio clips (< 1 minute)
- **Parallel processing** across three dimensions
- **Memory efficient** with streaming-friendly design
- **No external API calls** for core functionality
- **Offline capable** for all analysis except online speech recognition

## 🔬 Technical Details

### Musical Analysis Algorithms
- **Onset Detection**: Librosa's spectral flux method
- **Tempo Estimation**: Inter-onset interval analysis
- **Key Detection**: Chromagram peak analysis
- **Chord Analysis**: Harmonic content mapping

### Vocal Analysis Methods
- **Voice Activity**: Spectral centroid + rolloff heuristics
- **Speech Recognition**: Google Speech API (free tier)
- **Sentiment Analysis**: TextBlob VADER sentiment
- **Pitch Tracking**: Librosa piptrack algorithm

### Spectral Analysis Techniques
- **FFT Analysis**: Numpy-based frequency domain analysis
- **Harmonic-Percussive Separation**: Librosa HPSS
- **Spectral Features**: Librosa feature extraction
- **Dynamic Range**: RMS energy temporal analysis

## 🤝 Contributing

This is a free and open-source project! Contributions welcome:

1. **Feature requests** - Suggest new analysis capabilities
2. **Bug reports** - Help us improve reliability  
3. **Performance optimizations** - Make it faster
4. **Documentation** - Help others use the system
5. **New algorithms** - Add cutting-edge analysis methods

## 📜 License

This project uses only free and open-source libraries:
- **MIT License** compatible
- **No proprietary dependencies**
- **No usage restrictions**
- **Commercial use allowed**

## 🆘 Troubleshooting

### Common Issues

**Speech recognition not working:**
```bash
# Check internet connection (required for Google Speech API)
# Or install offline alternative:
pip install vosk
```

**Audio format not supported:**
```bash
# Install ffmpeg for additional format support
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**Import errors:**
```bash
# Ensure virtual environment is activated
source audio_env/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

**Slow performance:**
```bash
# Use lower sample rate for faster processing
agent = AudioAnalysisAgent(sample_rate=16000)
```

## 🎵 Ready to Analyze Audio!

This system provides professional-grade audio analysis using only free, open-source tools. No API keys, no payment required, no limitations!

Perfect for:
- 🎵 Music producers and composers
- 🎤 Podcasters and content creators  
- 🔬 Researchers and students
- 🎧 Audio enthusiasts
- 🤖 Developers building audio applications

**Get started now:**
```bash
./install.sh
source audio_env/bin/activate  
python audio_agent_lite.py
```
