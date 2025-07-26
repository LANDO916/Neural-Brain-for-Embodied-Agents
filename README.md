# Audio Analysis AI Agent

A comprehensive multi-layer audio analysis tool that performs deep audio deconstruction and generates video prompts for AI video generation. This React-based application uses the Web Audio API for real-time audio visualization and analysis.

## 🎯 Features

### 🎵 Audio Upload & Playback
- Support for multiple audio formats (MP3, WAV, etc.)
- Real-time audio waveform visualization
- Play/pause controls with visual feedback

### 📊 Multi-Layer Analysis

#### Layer 1: Parallel Deconstruction
- **Musical & Structural Analysis**
  - BPM detection with visual circular progress
  - Tempo analysis (Allegro, Moderato, Andante)
  - Genre classification
  - Key detection with Circle of Fifths visualization
  - Chord progression analysis
  
- **Lyrical & Vocal Analysis**
  - Live transcription simulation
  - Language detection
  - Sentiment analysis
  - Theme extraction
  - Vocal performance metrics (pitch accuracy, timbre)
  
- **Sonic & Spectral Analysis**
  - Real-time frequency spectrum visualization
  - Spectrogram with color-coded frequency analysis
  - Oscilloscope display
  - Instrumentation detection
  - Dynamic range analysis
  - Spatial audio characteristics

#### Layer 2: Synthesis & Application
- **Visual Synthesis**
  - Waveform pattern analysis
  - Spectrogram generation
  - Stereo phase correlation
  
- **Emotion Analysis**
  - Valence and arousal detection
  - Mood classification
  - Emotional arc mapping
  - Cross-modal sentiment correlation
  
- **Quality Assessment**
  - Clipping detection
  - Distortion analysis
  - Noise floor measurement
  - Compression assessment
  
- **Adaptive Learning**
  - Trend pattern recognition
  - Neural network optimization
  - User preference integration

#### Layer 3: Video Prompt Generation
- **Narrative Construction**
  - Story arc development
  - Character design suggestions
  - Emotional journey mapping
  - Symbolic element recommendations
  
- **Visual Styling**
  - Aesthetic theme generation
  - Lighting recommendations
  - Camera movement suggestions
  - Texture and material specifications
  
- **Technical Execution**
  - Scene composition guidelines
  - AI prompt engineering
  - Shot list generation
  - Quality optimization feedback

### 🎨 Real-Time Visualizations

1. **Main Waveform Display** - Real-time audio waveform
2. **Frequency Spectrum** - Dynamic bar chart showing frequency distribution
3. **Spectrogram** - Scrolling time-frequency heatmap
4. **Oscilloscope** - Time-domain signal display
5. **Interactive BPM Meter** - Circular progress with beat visualization
6. **Genre Selector** - Animated genre classification display
7. **Circle of Fifths** - Musical key visualization
8. **Chord Progression Flow** - Animated harmonic analysis

## 🚀 Quick Start

### Option 1: Direct HTML File
1. Open `index.html` in any modern web browser
2. The application will load with all dependencies from CDNs
3. Upload an audio file and start analyzing

### Option 2: React Component
1. Copy `AudioAnalysisAgent.jsx` into your React project
2. Install required dependencies:
   ```bash
   npm install lucide-react
   ```
3. Import and use the component:
   ```javascript
   import AudioAnalysisAgent from './AudioAnalysisAgent';
   
   function App() {
     return <AudioAnalysisAgent />;
   }
   ```

## 🎛️ Usage Instructions

1. **Upload Audio File**
   - Click the upload area or drag and drop an audio file
   - Supported formats: MP3, WAV, OGG, and other browser-supported formats

2. **Start Playback**
   - Click the play button to begin audio playback
   - Real-time visualizations will activate automatically
   - Live transcription will begin displaying analysis results

3. **Run Analysis**
   - Click "Start Analysis" to begin the multi-layer analysis process
   - Watch the progress indicator as each layer completes
   - Results will populate in real-time for each analysis layer

4. **Review Results**
   - Explore the detailed analysis results across all three layers
   - Use the generated video prompts for AI video creation tools
   - Musical analysis provides insights for remix or composition work

## 🔧 Technical Requirements

- Modern web browser with Web Audio API support
- JavaScript enabled
- Audio file access permissions
- Recommended: Chrome, Firefox, Safari, or Edge (latest versions)

## 🎪 Demo Features

The application includes simulated analysis results that demonstrate:
- Realistic BPM detection (80-160 range)
- Random but plausible musical key and chord progressions
- Emotion analysis with valence/arousal scoring
- Professional video prompt generation
- Live transcription with musical structure detection

## 🎨 Styling & Customization

The interface uses:
- **Tailwind CSS** for responsive styling
- **Dark theme** optimized for audio production environments
- **Gradient accents** for visual appeal
- **Custom animations** for smooth transitions
- **Lucide React icons** for consistent iconography

## 📱 Responsive Design

The application is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile devices (with touch-friendly controls)

## 🛠️ Browser Compatibility

- ✅ Chrome 66+
- ✅ Firefox 60+
- ✅ Safari 14+
- ✅ Edge 79+
- ❌ Internet Explorer (not supported)

## 🎵 Audio Format Support

Supported audio formats depend on browser capabilities:
- **MP3** - Universal support
- **WAV** - Universal support
- **OGG** - Firefox, Chrome
- **AAC/M4A** - Safari, Chrome, Edge
- **FLAC** - Chrome, Firefox (recent versions)

## 🔮 Future Enhancements

Potential features for future development:
- Real AI-powered audio analysis
- Integration with music streaming APIs
- Export capabilities for analysis results
- Collaborative analysis sharing
- Advanced machine learning models
- MIDI file analysis support
- Batch processing capabilities

## 📄 License

This project is provided as-is for educational and demonstration purposes. Feel free to modify and adapt for your specific needs.

## 🙋‍♂️ Support

For questions or issues:
1. Check browser console for error messages
2. Ensure audio file permissions are granted
3. Try different audio file formats if upload fails
4. Verify browser compatibility with Web Audio API

## 🎉 Acknowledgments

Built with:
- React 18
- Tailwind CSS
- Lucide React Icons
- Web Audio API
- HTML5 Canvas API
