import React, { useState, useRef, useEffect } from 'react';
import { Upload, Music, Mic, Activity, Brain, Video, Sparkles, Settings, Play, Pause, BarChart3, FileAudio, Layers } from 'lucide-react';

const AudioAnalysisAgent = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [analysisResults, setAnalysisResults] = useState({
    layer1: {
      musical: null,
      lyrical: null,
      sonic: null
    },
    layer2: {
      visual: null,
      emotion: null,
      quality: null,
      adaptive: null
    },
    layer3: {
      narrative: null,
      visual: null,
      technical: null
    }
  });
  
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const spectrumCanvasRef = useRef(null);
  const spectrogramCanvasRef = useRef(null);
  const oscilloscopeCanvasRef = useRef(null);
  const animationRef = useRef(null);
  const spectrogramAnimationRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const spectrogramDataRef = useRef([]);
  const [transcriptionText, setTranscriptionText] = useState('');
  const [transcriptionComplete, setTranscriptionComplete] = useState(false);

  // File upload handler
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('audio/')) {
      // Stop current playback if any
      if (audioRef.current && isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      }
      
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setTranscriptionText('');
      setTranscriptionComplete(false);
      setAnalysisResults({
        layer1: { musical: null, lyrical: null, sonic: null },
        layer2: { visual: null, emotion: null, quality: null, adaptive: null },
        layer3: { narrative: null, visual: null, technical: null }
      });
    }
  };

  // Initialize audio context and analyser
  const initAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      
      if (audioRef.current) {
        const source = audioContextRef.current.createMediaElementSource(audioRef.current);
        source.connect(analyserRef.current);
        analyserRef.current.connect(audioContextRef.current.destination);
      }
    }
  };

  // Visualize audio waveform
  const visualize = () => {
    if (!analyserRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyserRef.current.getByteTimeDomainData(dataArray);
      
      ctx.fillStyle = 'rgb(17, 24, 39)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgb(59, 130, 246)';
      ctx.beginPath();
      
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    
    draw();
  };

  // Frequency Spectrum Visualizer
  const visualizeSpectrum = () => {
    if (!analyserRef.current || !spectrumCanvasRef.current) return;
    
    const canvas = spectrumCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isPlaying) return;
      requestAnimationFrame(draw);
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      ctx.fillStyle = 'rgb(17, 24, 39)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
        
        // Create gradient bars
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, 'rgb(147, 51, 234)'); // purple
        gradient.addColorStop(0.5, 'rgb(59, 130, 246)'); // blue
        gradient.addColorStop(1, 'rgb(34, 197, 94)'); // green
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    };
    
    draw();
  };

  // Spectrogram Visualizer
  const visualizeSpectrogram = () => {
    if (!analyserRef.current || !spectrogramCanvasRef.current) return;
    
    const canvas = spectrogramCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isPlaying) return;
      spectrogramAnimationRef.current = requestAnimationFrame(draw);
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Shift existing image to the left
      const imageData = ctx.getImageData(1, 0, canvas.width - 1, canvas.height);
      ctx.putImageData(imageData, 0, 0);
      
      // Draw new column on the right
      for (let i = 0; i < bufferLength; i++) {
        const value = dataArray[i];
        const y = Math.floor((i / bufferLength) * canvas.height);
        const height = Math.ceil(canvas.height / bufferLength);
        
        // Color mapping: blue (low) -> green -> yellow -> red (high)
        let r, g, b;
        if (value < 85) {
          r = 0;
          g = 0;
          b = value * 3;
        } else if (value < 170) {
          r = 0;
          g = (value - 85) * 3;
          b = 255 - (value - 85) * 3;
        } else {
          r = (value - 170) * 3;
          g = 255;
          b = 0;
        }
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(canvas.width - 1, canvas.height - y - height, 1, height);
      }
    };
    
    draw();
  };

  // Oscilloscope Visualizer
  const visualizeOscilloscope = () => {
    if (!analyserRef.current || !oscilloscopeCanvasRef.current) return;
    
    const canvas = oscilloscopeCanvasRef.current;
    const ctx = canvas.getContext('2d');
    analyserRef.current.fftSize = 2048;
    const bufferLength = analyserRef.current.fftSize;
    const dataArray = new Float32Array(bufferLength);
    
    const draw = () => {
      if (!isPlaying) return;
      requestAnimationFrame(draw);
      
      analyserRef.current.getFloatTimeDomainData(dataArray);
      
      ctx.fillStyle = 'rgb(17, 24, 39)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgb(34, 197, 94)';
      ctx.beginPath();
      
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const y = (dataArray[i] * canvas.height / 2) + canvas.height / 2;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        
        x += sliceWidth;
      }
      
      ctx.stroke();
      
      // Draw center line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    
    draw();
  };

  // Play/pause audio
  const togglePlayPause = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      initAudioContext();
      
      // Clear spectrogram canvas on new play
      if (spectrogramCanvasRef.current) {
        const ctx = spectrogramCanvasRef.current.getContext('2d');
        ctx.fillStyle = 'rgb(17, 24, 39)';
        ctx.fillRect(0, 0, spectrogramCanvasRef.current.width, spectrogramCanvasRef.current.height);
      }
      
      audioRef.current.play();
      visualize();
      visualizeSpectrum();
      visualizeSpectrogram();
      visualizeOscilloscope();
    }
    setIsPlaying(!isPlaying);
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (spectrogramAnimationRef.current) {
        cancelAnimationFrame(spectrogramAnimationRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Initialize canvases with dark background
  useEffect(() => {
    const canvases = [spectrumCanvasRef, spectrogramCanvasRef, oscilloscopeCanvasRef];
    canvases.forEach(canvasRef => {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.fillStyle = 'rgb(17, 24, 39)';
        ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    });
  }, [audioUrl]);

  // Handle audio ended event
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleEnded = () => {
      setIsPlaying(false);
    };

    audio.addEventListener('ended', handleEnded);
    return () => {
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioUrl]);

  // Simulate live transcription
  useEffect(() => {
    if (!isPlaying) return;

    const transcriptionPhrases = [
      "🎵 [0:00] Musical introduction detected...",
      "📊 Analyzing frequency spectrum: Strong bass presence at 80-120Hz",
      "🎹 [0:12] Instrumental passage with rising melody",
      "🎤 [0:24] Detecting vocal patterns... Human voice identified",
      "📝 Speech clarity: 92% | Background noise: -45dB",
      "🎼 [0:36] Verse structure identified - 8 bar pattern",
      "🔊 Harmonic analysis: I - V - vi - IV progression detected",
      "💫 [0:48] Energy surge detected → Transition to chorus",
      "🎵 [1:00] Chorus section beginning with doubled vocals",
      "📈 Dynamic range expanding: +6dB increase measured",
      "🎸 [1:12] Instrumental solo detected - Guitar lead",
      "🌊 [1:24] Bridge section with key modulation to relative minor",
      "🎹 Rhythmic variation: Syncopated pattern observed",
      "✨ [1:36] Vocal harmonies detected - 3-part arrangement",
      "📉 [1:48] Energy decreasing → Approaching outro",
      "🎵 [2:00] Outro section with fade effect..."
    ];

    let currentIndex = 0;
    setTranscriptionText(''); // Clear on new playback
    setTranscriptionComplete(false);

    const interval = setInterval(() => {
      if (currentIndex < transcriptionPhrases.length && isPlaying) {
        setTranscriptionText(prev => {
          if (prev) return prev + '\n' + transcriptionPhrases[currentIndex];
          return transcriptionPhrases[currentIndex];
        });
        currentIndex++;
        
        // Auto-scroll to bottom
        const scrollElement = document.querySelector('.transcription-scroll');
        if (scrollElement) {
          scrollElement.scrollTop = scrollElement.scrollHeight;
        }
      } else {
        clearInterval(interval);
        if (currentIndex >= transcriptionPhrases.length) {
          setTranscriptionComplete(true);
        }
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [isPlaying]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Audio Analysis AI Agent
          </h1>
          <p className="text-gray-400">Multi-layer audio deconstruction and video prompt generation</p>
        </div>

        {/* File Upload */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer hover:bg-gray-700 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="w-10 h-10 mb-3 text-gray-400" />
              <p className="mb-2 text-sm text-gray-400">
                <span className="font-semibold">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">Audio files (MP3, WAV, etc.)</p>
            </div>
            <input type="file" className="hidden" accept="audio/*" onChange={handleFileUpload} />
          </label>
          
          {audioFile && (
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <FileAudio className="w-6 h-6 text-blue-400" />
                <span className="text-sm">{audioFile.name}</span>
              </div>
              <button
                onClick={() => {/* Analysis function will be added */}}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Start Analysis
              </button>
            </div>
          )}
        </div>

        {/* Audio Player & Visualizer */}
        {audioUrl && (
          <div className="bg-gray-800 rounded-lg p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">Audio Playback</h3>
              <button
                onClick={togglePlayPause}
                className="p-2 bg-gray-700 hover:bg-gray-600 rounded-full transition-colors"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
            </div>
            <audio ref={audioRef} src={audioUrl} className="hidden" />
            <canvas
              ref={canvasRef}
              width={800}
              height={200}
              className="w-full h-32 bg-gray-900 rounded"
            />
          </div>
        )}

        {/* Analysis results will be added here */}
      </div>
      
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(31, 41, 55, 0.5);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(107, 114, 128, 0.5);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(107, 114, 128, 0.7);
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(5px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out forwards;
        }
        @keyframes pulse {
          0% {
            transform: scale(0.9);
            opacity: 0.8;
          }
          50% {
            transform: scale(1.1);
            opacity: 0.3;
          }
          100% {
            transform: scale(0.9);
            opacity: 0.8;
          }
        }
      `}</style>
    </div>
  );
};

export default AudioAnalysisAgent;