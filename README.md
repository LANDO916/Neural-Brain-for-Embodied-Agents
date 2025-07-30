# Audio Analysis Agent - Layer 1: Parallel Deconstruction

A comprehensive audio analysis system that analyzes audio across three distinct dimensions simultaneously, as described in the "Layer 1: Parallel Deconstruction" diagram.

## 🎵 Features

### 1. Musical & Structural Analysis
- **BPM & Tempo Mapper**: Detects beats per minute and tempo variations
- **Groove & Swing Analyzer**: Analyzes syncopation and micro-timing patterns
- **Tonal Center Identifier**: Determines key and mode (major/minor)
- **Harmonic Journey Mapper**: Transcribes chord progressions and key changes
- **Genre & Style Agent**: Assigns genre tags based on stylistic cues

### 2. Lyrical & Vocal Analysis
- **Speech-to-Text Engine**: Converts vocals to text (requires whisper)
- **Poetic Structure Analyzer**: Identifies rhyme schemes and literary devices
- **Lexical Sentiment Scorer**: Analyzes emotional tone of lyrics
- **Thematic Extractor**: Identifies overarching topics and themes
- **Pitch & Melody Contour Analyzer**: Maps vocal melodic performance
- **Vocal Timbre Identifier**: Characterizes voice tone characteristics

### 3. Sonic & Spectral Analysis
- **Source Separator**: Isolates instrument stems (basic implementation)
- **Timbre Fingerprinter**: Analyzes harmonic overtones for instrument identification
- **Loudness Meter (LUFS)**: Calculates integrated and short-term loudness
- **Spectral Power Monitor**: Tracks energy across frequency bands
- **Stereo Image Analyzer**: Measures stereo width and phase correlation
- **Reverb Impulse Modeler**: Estimates room size and decay characteristics

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Basic audio processing libraries

### Installation

1. **Install dependencies** (choose one method):

   **Option A: Using pip with system packages (if you have root access):**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   **Option B: Using pip with --break-system-packages (if no root access):**
   ```bash
   pip install --break-system-packages -r requirements.txt
   ```

   **Option C: Minimal installation (core functionality only):**
   ```bash
   pip install --break-system-packages librosa numpy scipy soundfile
   ```

2. **Test the installation:**
   ```bash
   python test_audio_analysis.py
   ```

### Usage

#### Basic Usage
```bash
python audio_analysis_agent.py your_audio_file.wav
```

#### Example Output
```
Analyzing audio file: song.wav
==================================================
1. Musical & Structural Analysis...
2. Lyrical & Vocal Analysis...
3. Sonic & Spectral Analysis...

==================================================
ANALYSIS COMPLETE
==================================================

SUMMARY:
--------------------
BPM: 120.5
Key: C major
Loudness: -12.3 dB

Detailed results saved to: song_analysis.json
```

#### Supported Audio Formats
- WAV (recommended)
- MP3 (requires additional codecs)
- FLAC
- OGG
- Most formats supported by librosa

## 📊 Output Format

The agent generates a comprehensive JSON report with the following structure:

```json
{
  "musical_structural": {
    "bpm_and_tempo": { "bpm": 120.5, "dynamic_tempo": 121.2 },
    "groove_and_swing": { "swing_score": 1.1, "syncopation": 0.05 },
    "tonal_center": { "key": "C", "mode": "major", "confidence": 0.85 },
    "harmonic_journey": { "num_chord_changes": 8, "harmonic_complexity": 0.3 },
    "genre_and_style": { "style_hint": "medium/rock" }
  },
  "lyrical_vocal": {
    "transcription": { "transcript": "...", "language": "en" },
    "poetic_structure": { "word_count": 150, "vocabulary_diversity": 0.7 },
    "sentiment": { "sentiment_score": 0.3, "overall_tone": "positive" },
    "themes": { "detected_themes": { "love": 5, "emotion": 3 } },
    "pitch_and_melody": { "pitch_range_semitones": 12.5 },
    "vocal_timbre": { "timbre_type": "bright", "brightness": 2500 }
  },
  "sonic_spectral": {
    "source_separation": { "harmonic_ratio": 0.6, "percussive_ratio": 0.4 },
    "timbre_fingerprint": { "fingerprint_dimensions": 23 },
    "loudness": { "loudness_db": -12.3, "dynamic_range_db": 15.7 },
    "spectral_power": { "spectral_balance": "balanced" },
    "stereo_image": { "stereo_width": 0.3, "stereo_type": "stereo" },
    "reverb_model": { "decay_time_seconds": 0.8, "estimated_room_size": "medium" }
  },
  "metadata": {
    "audio_file": "song.wav",
    "analysis_timestamp": "2024-01-01T12:00:00",
    "analysis_version": "1.0"
  }
}
```

## 🔧 Advanced Features

### Custom Analysis
```python
from audio_analysis_agent import AudioAnalyzer

analyzer = AudioAnalyzer()

# Run specific analysis
bpm_result = analyzer.bpm_and_tempo_mapper("song.wav")
key_result = analyzer.tonal_center_identifier("song.wav")

# Run complete analysis
result = analyzer.analyze_audio("song.wav")
```

### Batch Processing
```python
import os
from audio_analysis_agent import AudioAnalyzer

analyzer = AudioAnalyzer()
audio_files = [f for f in os.listdir('.') if f.endswith('.wav')]

for audio_file in audio_files:
    print(f"Analyzing {audio_file}...")
    result = analyzer.analyze_audio(audio_file)
    # Process results as needed
```

## 🛠️ Troubleshooting

### Common Issues

1. **ImportError: No module named 'librosa'**
   ```bash
   pip install --break-system-packages librosa
   ```

2. **Audio file not found**
   - Ensure the audio file path is correct
   - Check file permissions

3. **Analysis fails with error**
   - Check if the audio file is corrupted
   - Try with a different audio format (WAV recommended)
   - Ensure sufficient disk space for temporary files

4. **Slow performance**
   - Use shorter audio files for testing
   - Consider downsampling long files
   - Close other applications to free up memory

### Performance Tips

- **For large files**: Consider processing in chunks
- **For batch processing**: Use multiprocessing
- **For real-time analysis**: Use streaming audio input

## 📈 Future Enhancements

- [ ] Integration with trained genre classification models
- [ ] Advanced source separation using Spleeter
- [ ] Real-time analysis capabilities
- [ ] Web interface for easy interaction
- [ ] Support for video files with audio extraction
- [ ] Export to various formats (CSV, Excel, etc.)

## 🤝 Contributing

This is a working implementation of the audio analysis system described in the "Layer 1: Parallel Deconstruction" diagram. The system is designed to be:

- **Modular**: Each analysis component can be used independently
- **Extensible**: Easy to add new analysis methods
- **Robust**: Handles errors gracefully and provides fallbacks
- **Free**: Uses only open-source libraries and can run completely offline

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **librosa**: For audio processing capabilities
- **numpy/scipy**: For mathematical operations
- **soundfile**: For audio file I/O
- The original "Layer 1: Parallel Deconstruction" diagram that inspired this implementation
