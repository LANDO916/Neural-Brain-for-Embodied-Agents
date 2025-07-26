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

  // Clear transcription when audio stops
  useEffect(() => {
    if (!isPlaying) {
      // Keep transcription visible after stopping, only clear on new play
    }
  }, [isPlaying]);

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

  // Layer 1: Parallel Deconstruction Analysis
  const performLayer1Analysis = async () => {
    setCurrentLayer(1);
    
    // Simulate Musical & Structural Analysis
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer1: {
          ...prev.layer1,
          musical: {
            bpm: Math.floor(Math.random() * 80) + 80,
            timeSignature: '4/4',
            tempo: ['Allegro', 'Moderato', 'Andante'][Math.floor(Math.random() * 3)],
            genre: ['Pop', 'Rock', 'Electronic', 'Classical'][Math.floor(Math.random() * 4)],
            harmonyAgent: {
              key: ['C Major', 'G Major', 'A Minor', 'E Minor'][Math.floor(Math.random() * 4)],
              chords: ['I-V-vi-IV', 'ii-V-I', 'I-IV-V'][Math.floor(Math.random() * 3)],
              modulations: Math.floor(Math.random() * 3)
            },
            taxonomy: {
              primaryGenre: 'Electronic',
              subGenres: ['Ambient', 'Synthwave', 'Downtempo']
            }
          }
        }
      }));
    }, 1000);

    // Simulate Lyrical & Vocal Analysis
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer1: {
          ...prev.layer1,
          lyrical: {
            transcription: "Sample lyrics detected...",
            language: 'English',
            sentiment: {
              overall: 'Positive',
              scores: { positive: 0.7, negative: 0.1, neutral: 0.2 }
            },
            themes: ['Love', 'Journey', 'Hope'],
            vocalPerformance: {
              pitchAccuracy: 0.92,
              melodyContour: 'Ascending',
              timbre: 'Warm and breathy'
            }
          }
        }
      }));
    }, 1500);

    // Simulate Sonic & Spectral Analysis
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer1: {
          ...prev.layer1,
          sonic: {
            instrumentation: ['Synthesizer', 'Drums', 'Bass', 'Vocals'],
            timbre: {
              brightness: 0.7,
              warmth: 0.6,
              presence: 0.8
            },
            dynamics: {
              loudness: -14,
              dynamicRange: 8,
              peakLevel: -0.3
            },
            spatial: {
              stereoWidth: 0.85,
              depth: 'Moderate',
              reverb: 'Hall'
            }
          }
        }
      }));
    }, 2000);
  };

  // Layer 2: Synthesis & Application
  const performLayer2Analysis = async () => {
    setCurrentLayer(2);
    
    // Visual Synthesis
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer2: {
          ...prev.layer2,
          visual: {
            waveformPattern: 'Dynamic with clear peaks',
            spectrogramGenerator: 'Frequency-over-time heatmap generated',
            vectorscope: 'Stereo phase correlation: Good'
          }
        }
      }));
    }, 500);

    // Holistic Emotion Agent
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer2: {
          ...prev.layer2,
          emotion: {
            valence: 0.75,
            arousal: 0.6,
            mood: 'Uplifting and energetic',
            emotionalArc: 'Building from calm to euphoric',
            crossModal: 'Harmonic progression supports lyrical sentiment'
          }
        }
      }));
    }, 1000);

    // Quality & Integrity Agent
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer2: {
          ...prev.layer2,
          quality: {
            clipping: false,
            distortion: 0.02,
            noiseFloor: -60,
            compression: 'Moderate',
            phaseIssues: 'None detected'
          }
        }
      }));
    }, 1500);

    // Adaptive Learning Engine
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer2: {
          ...prev.layer2,
          adaptive: {
            trendSpotter: 'Synthwave revival pattern detected',
            modelRetrainer: 'Neural network updated with new examples',
            feedbackLoop: 'User preferences incorporated'
          }
        }
      }));
    }, 2000);
  };

  // Layer 3: Video Prompt Generation
  const performLayer3Analysis = async () => {
    setCurrentLayer(3);
    
    // Narrative Agents
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer3: {
          ...prev.layer3,
          narrative: {
            architect: "A journey through neon-lit cityscapes, following a lone protagonist discovering hidden digital realms",
            characters: "Cyberpunk explorer with glowing augmentations",
            emotionalBeat: "Rising tension → Discovery → Euphoric revelation",
            symbolism: "Digital butterflies representing transformation, neon rain as cleansing"
          }
        }
      }));
    }, 1000);

    // Visual Styling Agents
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer3: {
          ...prev.layer3,
          visual: {
            aesthetic: "Synthwave meets cyberpunk - vibrant neons, deep purples, electric blues",
            lighting: "Volumetric fog with colored rim lighting, dramatic shadows",
            movement: "Smooth camera glides with synchronized beat cuts",
            textures: "Glossy reflective surfaces, holographic overlays, particle effects"
          }
        }
      }));
    }, 1500);

    // Technical Execution Agents
    setTimeout(() => {
      setAnalysisResults(prev => ({
        ...prev,
        layer3: {
          ...prev.layer3,
          technical: {
            sceneComposition: "Wide establishing shots → Medium character focus → Close detail inserts",
            promptEngineer: "Cinematic, 8K, ray-traced reflections, volumetric lighting, depth of field",
            feedbackLoop: "Render quality optimized based on previous generations",
            blueprints: "Shot list: 12 scenes, 3-5 seconds each, matched to musical phrases"
          }
        }
      }));
    }, 2000);
  };

  // Main analysis pipeline
  const startAnalysis = async () => {
    if (!audioFile) return;
    
    setIsAnalyzing(true);
    setCurrentLayer(0);
    
    // Layer 1
    await performLayer1Analysis();
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Layer 2
    await performLayer2Analysis();
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Layer 3
    await performLayer3Analysis();
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    setIsAnalyzing(false);
    setCurrentLayer(0);
  };

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
                onClick={startAnalysis}
                disabled={isAnalyzing}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
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

        {/* Layer Progress */}
        {isAnalyzing && (
          <div className="bg-gray-800 rounded-lg p-6 mb-8">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold">Analysis Progress</h3>
              <Layers className="w-6 h-6 text-blue-400" />
            </div>
            <div className="space-y-4">
              {[1, 2, 3].map((layer) => (
                <div key={layer} className="flex items-center space-x-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    currentLayer >= layer ? 'bg-blue-600' : 'bg-gray-700'
                  }`}>
                    {layer}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium mb-1">
                      Layer {layer}: {layer === 1 ? 'Parallel Deconstruction' : layer === 2 ? 'Synthesis & Application' : 'Video Prompt Generation'}
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-1000"
                        style={{ width: currentLayer >= layer ? '100%' : '0%' }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Layer 1 Results */}
        {analysisResults.layer1.musical && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <span className="bg-blue-600 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">1</span>
              Layer 1: Parallel Deconstruction
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Musical & Structural Analysis */}
              <div className="bg-gray-800 rounded-lg p-6 flex flex-col min-h-[650px]">
                <div className="flex items-center mb-4">
                  <Music className="w-6 h-6 mr-2 text-blue-400" />
                  <h3 className="text-lg font-semibold">Musical & Structural</h3>
                </div>
                <div className="space-y-4">
                  {/* BPM Visualizer */}
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">BPM:</span>
                    <div className="flex items-center space-x-3">
                      <div className="relative w-24 h-24">
                        <svg className="w-full h-full -rotate-90">
                          <circle
                            cx="48"
                            cy="48"
                            r="40"
                            stroke="rgba(59, 130, 246, 0.2)"
                            strokeWidth="8"
                            fill="none"
                          />
                          <circle
                            cx="48"
                            cy="48"
                            r="40"
                            stroke="rgb(59, 130, 246)"
                            strokeWidth="8"
                            fill="none"
                            strokeDasharray={`${(analysisResults.layer1.musical.bpm / 200) * 251.2} 251.2`}
                            className="transition-all duration-500"
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="text-center">
                            <div className="text-2xl font-bold">{analysisResults.layer1.musical.bpm}</div>
                            <div className="text-xs text-gray-400">BPM</div>
                          </div>
                        </div>
                        {isPlaying && (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div 
                              className="w-16 h-16 rounded-full border-2 border-blue-400 opacity-20"
                              style={{
                                animation: `pulse ${60000 / analysisResults.layer1.musical.bpm}ms infinite`
                              }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Tempo Visualizer */}
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Tempo:</span>
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        <div className="w-32 bg-gray-900 rounded-full h-3 relative overflow-hidden">
                          <div 
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full transition-all duration-500"
                            style={{
                              width: `${
                                analysisResults.layer1.musical.tempo === 'Allegro' ? '100%' :
                                analysisResults.layer1.musical.tempo === 'Moderato' ? '60%' :
                                '30%'
                              }`
                            }}
                          />
                        </div>
                        <div className="flex justify-between mt-1 text-xs text-gray-600">
                          <span>Slow</span>
                          <span>Fast</span>
                        </div>
                      </div>
                      <span className="text-sm font-medium min-w-[80px] text-right">{analysisResults.layer1.musical.tempo}</span>
                    </div>
                  </div>

                  {/* Genre Visualizer */}
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Genre:</span>
                    <div className="flex items-center space-x-3">
                      <div className="relative w-24 h-24">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                            <Music className="w-4 h-4 text-gray-400" />
                          </div>
                          {['Pop', 'Rock', 'Electronic', 'Classical'].map((genre, index) => {
                            const angle = (index * 90) - 90;
                            const isActive = genre === analysisResults.layer1.musical.genre;
                            return (
                              <div
                                key={genre}
                                className={`absolute w-16 h-8 rounded-full flex items-center justify-center text-xs font-medium transition-all duration-300 ${
                                  isActive ? 'bg-blue-600 scale-110 shadow-lg shadow-blue-600/50' : 'bg-gray-700 scale-90 opacity-70'
                                }`}
                                style={{
                                  transform: `rotate(${angle}deg) translateY(-35px) rotate(-${angle}deg)`
                                }}
                              >
                                {genre.substring(0, 3).toUpperCase()}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      <span className="text-sm font-medium min-w-[80px] text-right">{analysisResults.layer1.musical.genre}</span>
                    </div>
                  </div>

                  {/* Key Visualizer */}
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Key:</span>
                    <div className="flex items-center space-x-3">
                      <div className="relative w-24 h-24">
                        <svg className="w-full h-full">
                          {/* Circle of Fifths visualization */}
                          <circle cx="48" cy="48" r="40" fill="none" stroke="rgba(59, 130, 246, 0.2)" strokeWidth="2" />
                          {/* Center circle */}
                          <circle cx="48" cy="48" r="12" fill="rgba(59, 130, 246, 0.1)" />
                          <text x="48" y="48" textAnchor="middle" dominantBaseline="middle" className="text-xs fill-blue-400 font-bold">
                            5ths
                          </text>
                          {['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'].map((note, index) => {
                            const angle = (index * 30) - 90;
                            const x = 48 + 32 * Math.cos(angle * Math.PI / 180);
                            const y = 48 + 32 * Math.sin(angle * Math.PI / 180);
                            const keyNote = analysisResults.layer1.musical.harmonyAgent.key.split(' ')[0];
                            const isActive = keyNote === note || (keyNote === 'F' && note === 'F');
                            return (
                              <g key={note}>
                                <circle
                                  cx={x}
                                  cy={y}
                                  r={isActive ? 8 : 6}
                                  fill={isActive ? 'rgb(59, 130, 246)' : 'rgba(107, 114, 128, 0.5)'}
                                  className="transition-all duration-300"
                                />
                                <text
                                  x={x}
                                  y={y}
                                  textAnchor="middle"
                                  dominantBaseline="middle"
                                  className="text-xs fill-white font-medium"
                                  style={{ fontSize: isActive ? '10px' : '8px' }}
                                >
                                  {note}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      </div>
                      <span className="text-sm font-medium min-w-[80px] text-right">{analysisResults.layer1.musical.harmonyAgent.key}</span>
                    </div>
                  </div>

                  {/* Chord Progression Visualizer */}
                  <div className="flex flex-col space-y-2">
                    <span className="text-gray-400 text-sm">Chord Progression:</span>
                    <div className="flex items-center justify-center flex-wrap gap-2">
                      {analysisResults.layer1.musical.harmonyAgent.chords.split('-').map((chord, index) => (
                        <div key={index} className="flex items-center">
                          <div className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 border border-blue-600/30 ${
                            isPlaying ? 'animate-pulse' : ''
                          }`}
                          style={{
                            backgroundColor: `rgba(59, 130, 246, ${0.2 + (index * 0.15)})`,
                            animationDelay: `${index * 200}ms`
                          }}>
                            {chord}
                          </div>
                          {index < analysisResults.layer1.musical.harmonyAgent.chords.split('-').length - 1 && (
                            <span className="text-gray-500 mx-1">→</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Lyrical & Vocal Analysis */}
              {analysisResults.layer1.lyrical && (
                <div className="bg-gray-800 rounded-lg p-6 flex flex-col min-h-[650px]">
                  <div className="flex items-center mb-4">
                    <Mic className="w-6 h-6 mr-2 text-purple-400" />
                    <h3 className="text-lg font-semibold">Lyrical & Vocal</h3>
                  </div>
                  <div className="space-y-2 text-sm mb-4">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Language:</span>
                      <span>{analysisResults.layer1.lyrical.language}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sentiment:</span>
                      <span>{analysisResults.layer1.lyrical.sentiment.overall}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Themes:</span>
                      <span>{analysisResults.layer1.lyrical.themes.join(', ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pitch Accuracy:</span>
                      <span>{(analysisResults.layer1.lyrical.vocalPerformance.pitchAccuracy * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Timbre:</span>
                      <span>{analysisResults.layer1.lyrical.vocalPerformance.timbre}</span>
                    </div>
                  </div>
                  
                  {/* Live Transcription Panel */}
                  <div className="flex-1 bg-gray-900 rounded-lg p-4 overflow-hidden border border-gray-700 min-h-[240px]">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm text-gray-300 font-semibold flex items-center">
                        <span className="mr-2">📝</span> Live Transcription
                      </h4>
                      {isPlaying && (
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                          <span className="text-xs text-red-500 font-medium">LIVE</span>
                        </div>
                      )}
                      {!isPlaying && transcriptionComplete && (
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span className="text-xs text-green-500 font-medium">COMPLETE</span>
                        </div>
                      )}
                    </div>
                    <div className="h-56 overflow-y-auto pr-2 custom-scrollbar transcription-scroll">
                      {transcriptionText ? (
                        <div className="space-y-1">
                          {transcriptionText.split('\n').map((line, index) => (
                            <div
                              key={index}
                              className="text-xs text-gray-300 font-mono animate-fadeIn"
                              style={{ animationDelay: `${index * 0.1}s` }}
                            >
                              {line}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-gray-500 italic">
                          {audioUrl ? 'Play audio to begin transcription...' : 'Upload audio file to enable transcription'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Sonic & Spectral Analysis */}
              {analysisResults.layer1.sonic && (
                <div className="bg-gray-800 rounded-lg p-6 flex flex-col min-h-[650px]">
                  <div className="flex items-center mb-4">
                    <Activity className="w-6 h-6 mr-2 text-green-400" />
                    <h3 className="text-lg font-semibold">Sonic & Spectral</h3>
                  </div>
                  
                  {/* Spectral Visualizers */}
                  {audioUrl && (
                    <div className="mb-4">
                      {!isPlaying && (
                        <p className="text-xs text-gray-500 mb-2 italic">Play audio to see real-time spectral analysis</p>
                      )}
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <h4 className="text-xs text-gray-400 mb-2">Frequency Spectrum</h4>
                          <canvas
                            ref={spectrumCanvasRef}
                            width={200}
                            height={100}
                            className="w-full bg-gray-900 rounded"
                          />
                        </div>
                        <div>
                          <h4 className="text-xs text-gray-400 mb-2">Spectrogram</h4>
                          <canvas
                            ref={spectrogramCanvasRef}
                            width={200}
                            height={100}
                            className="w-full bg-gray-900 rounded"
                          />
                        </div>
                        <div>
                          <h4 className="text-xs text-gray-400 mb-2">Oscilloscope</h4>
                          <canvas
                            ref={oscilloscopeCanvasRef}
                            width={200}
                            height={100}
                            className="w-full bg-gray-900 rounded"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-400">Instruments:</span>
                      <div className="mt-1">{analysisResults.layer1.sonic.instrumentation.join(', ')}</div>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Loudness:</span>
                      <span>{analysisResults.layer1.sonic.dynamics.loudness} LUFS</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Dynamic Range:</span>
                      <span>{analysisResults.layer1.sonic.dynamics.dynamicRange} LU</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Stereo Width:</span>
                      <span>{(analysisResults.layer1.sonic.spatial.stereoWidth * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Reverb:</span>
                      <span>{analysisResults.layer1.sonic.spatial.reverb}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Layer 2 Results */}
        {analysisResults.layer2.visual && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <span className="bg-purple-600 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">2</span>
              Layer 2: Synthesis & Application
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Visual Synthesis & Emotion */}
              <div className="bg-gray-800 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <Sparkles className="w-6 h-6 mr-2 text-yellow-400" />
                  <h3 className="text-lg font-semibold">Visual & Emotional Synthesis</h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-400">Waveform Pattern:</span>
                    <p className="mt-1">{analysisResults.layer2.visual.waveformPattern}</p>
                  </div>
                  {analysisResults.layer2.emotion && (
                    <>
                      <div>
                        <span className="text-gray-400">Overall Mood:</span>
                        <p className="mt-1">{analysisResults.layer2.emotion.mood}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Emotional Arc:</span>
                        <p className="mt-1">{analysisResults.layer2.emotion.emotionalArc}</p>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Valence:</span>
                        <span>{(analysisResults.layer2.emotion.valence * 100).toFixed(0)}% positive</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Quality & Adaptive Learning */}
              <div className="bg-gray-800 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <Brain className="w-6 h-6 mr-2 text-pink-400" />
                  <h3 className="text-lg font-semibold">Quality & Adaptive Learning</h3>
                </div>
                <div className="space-y-3 text-sm">
                  {analysisResults.layer2.quality && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Clipping Detected:</span>
                        <span>{analysisResults.layer2.quality.clipping ? 'Yes' : 'No'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Distortion:</span>
                        <span>{(analysisResults.layer2.quality.distortion * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Noise Floor:</span>
                        <span>{analysisResults.layer2.quality.noiseFloor} dB</span>
                      </div>
                    </>
                  )}
                  {analysisResults.layer2.adaptive && (
                    <div>
                      <span className="text-gray-400">Trend Detected:</span>
                      <p className="mt-1">{analysisResults.layer2.adaptive.trendSpotter}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Layer 3 Results */}
        {analysisResults.layer3.narrative && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <span className="bg-green-600 rounded-full w-8 h-8 flex items-center justify-center mr-3 text-sm">3</span>
              Layer 3: Video Prompt Generation
            </h2>
            <div className="space-y-6">
              {/* Core Narrative */}
              <div className="bg-gray-800 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <Video className="w-6 h-6 mr-2 text-red-400" />
                  <h3 className="text-lg font-semibold">Core Narrative</h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-400 font-medium">Story Arc:</span>
                    <p className="mt-1">{analysisResults.layer3.narrative.architect}</p>
                  </div>
                  <div>
                    <span className="text-gray-400 font-medium">Characters:</span>
                    <p className="mt-1">{analysisResults.layer3.narrative.characters}</p>
                  </div>
                  <div>
                    <span className="text-gray-400 font-medium">Emotional Journey:</span>
                    <p className="mt-1">{analysisResults.layer3.narrative.emotionalBeat}</p>
                  </div>
                  <div>
                    <span className="text-gray-400 font-medium">Symbolism:</span>
                    <p className="mt-1">{analysisResults.layer3.narrative.symbolism}</p>
                  </div>
                </div>
              </div>

              {/* Visual Styling */}
              {analysisResults.layer3.visual && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <div className="flex items-center mb-4">
                    <Settings className="w-6 h-6 mr-2 text-cyan-400" />
                    <h3 className="text-lg font-semibold">Visual Styling</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400 font-medium">Aesthetic:</span>
                      <p className="mt-1">{analysisResults.layer3.visual.aesthetic}</p>
                    </div>
                    <div>
                      <span className="text-gray-400 font-medium">Lighting:</span>
                      <p className="mt-1">{analysisResults.layer3.visual.lighting}</p>
                    </div>
                    <div>
                      <span className="text-gray-400 font-medium">Camera Movement:</span>
                      <p className="mt-1">{analysisResults.layer3.visual.movement}</p>
                    </div>
                    <div>
                      <span className="text-gray-400 font-medium">Textures:</span>
                      <p className="mt-1">{analysisResults.layer3.visual.textures}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Technical Execution */}
              {analysisResults.layer3.technical && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <div className="flex items-center mb-4">
                    <BarChart3 className="w-6 h-6 mr-2 text-orange-400" />
                    <h3 className="text-lg font-semibold">Technical Execution</h3>
                  </div>
                  <div className="space-y-3 text-sm">
                    <div>
                      <span className="text-gray-400 font-medium">Scene Composition:</span>
                      <p className="mt-1">{analysisResults.layer3.technical.sceneComposition}</p>
                    </div>
                    <div>
                      <span className="text-gray-400 font-medium">Prompt Engineering:</span>
                      <p className="mt-1 font-mono text-xs bg-gray-900 p-2 rounded">
                        {analysisResults.layer3.technical.promptEngineer}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-400 font-medium">Shot Blueprint:</span>
                      <p className="mt-1">{analysisResults.layer3.technical.blueprints}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Complete Analysis Output */}
              {analysisResults.layer3.technical && (
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-center">
                  <h3 className="text-xl font-bold mb-2">Analysis Complete!</h3>
                  <p className="text-sm opacity-90">
                    Your audio has been fully analyzed across all three layers. 
                    The generated video prompts are ready for AI video generation.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
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