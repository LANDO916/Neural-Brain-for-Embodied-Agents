# Orpheum Dream Engine - Project Structure

## Overview
The Orpheum Dream Engine is a complete audio-to-visual dream generator with a minimal, mysterious interface powered by LLMs and real-time 3D visualization.

## Project Structure

```
orpheum-dream-engine/
├── orpheum/                    # Core Python package
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Main entry point (enables python -m orpheum)
│   └── engine.py              # Core OrpheumEngine class
├── web/                       # Web interface
│   ├── templates/
│   │   └── index.html         # Main HTML interface
│   └── static/
│       ├── css/
│       │   └── orpheum.css    # Minimal mysterious styles
│       ├── js/
│       │   ├── orpheum.js     # Main application logic
│       │   └── visualizer.js  # Three.js visualization
│       └── lib/
│           ├── three.min.js   # Three.js library
│           └── socket.io.min.js # Socket.IO library
├── scripts/
│   └── setup.sh              # Setup script
├── uploads/                   # Audio file uploads
├── outputs/                   # Generated outputs
├── workflows/                 # Processing workflows
├── logs/                      # Application logs
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── .env.example              # Environment configuration template
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
└── README.md                 # Project documentation
```

## Quick Start

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Engine**:
   ```bash
   python -m orpheum.engine
   ```

4. **Access Interface**:
   Open http://localhost:5000 in your browser

## Features

- **Minimal Interface**: Clean, mysterious UI with neural activity indicators
- **Audio Analysis**: Real-time feature extraction with librosa
- **LLM Integration**: GPT-4o powered interpretation of audio features
- **3D Visualization**: Three.js-based abstract visual generation
- **WebSocket Communication**: Real-time state updates via Socket.IO
- **Docker Support**: Containerized deployment ready

## Architecture Components

### Core Engine (`orpheum/engine.py`)
- Flask web application with Socket.IO
- Audio processing with librosa
- OpenAI GPT-4o integration
- State management and updates

### Web Interface (`web/`)
- Minimal HTML template
- Real-time JavaScript application
- Three.js 3D visualization
- Socket.IO client communication

### Processing Pipeline
1. **Audio Upload**: User selects audio file
2. **Feature Extraction**: Tempo, spectral centroid, chroma, energy
3. **LLM Interpretation**: GPT-4o analyzes features
4. **State Update**: Emotion, valence, arousal calculation
5. **Visualization**: Real-time 3D scene updates

## Technologies Used

- **Backend**: Python, Flask, Flask-SocketIO, librosa, OpenAI
- **Frontend**: HTML5, CSS3, JavaScript, Three.js, Socket.IO
- **Audio Processing**: librosa, numpy, scipy
- **AI/ML**: OpenAI GPT-4o API
- **Deployment**: Docker, Docker Compose

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ORPHEUM_HOST`: Server host (default: 0.0.0.0)
- `ORPHEUM_PORT`: Server port (default: 5000)
- `SECRET_KEY`: Flask secret key

## Development

The project follows a modular architecture with clear separation between:
- Audio processing and analysis
- LLM-powered interpretation
- Real-time web interface
- 3D visualization engine

All components communicate via the central OrpheumEngine orchestrator.