"""
Orpheum Dream Engine - Core System
Minimal, mysterious audio-to-visual dream generator
"""

import os
import json
import asyncio
import numpy as np
import librosa
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from openai import OpenAI
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('orpheum')

class OrpheumEngine:
    """Main orchestrator for the dream engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_client = OpenAI(api_key=config['openai_api_key'])
        self.current_state = {
            'emotion': 'neutral',
            'valence': 0.0,
            'arousal': 0.0,
            'confidence': 0.0,
            'active_node': 'INIT',
            'timestamp': datetime.now().isoformat()
        }
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        static_folder='../web/static',
                        template_folder='../web/templates')
        self.app.config['SECRET_KEY'] = config.get('secret_key', 'orpheum-secret')
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._setup_routes()
        logger.info("Orpheum Dream Engine initialized")
        
    def _setup_routes(self):
        """Setup web routes"""
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/process', methods=['POST'])
        async def process_audio():
            try:
                if 'audio' not in request.files:
                    return jsonify({'error': 'No audio file'}), 400
                    
                audio_file = request.files['audio']
                
                # Process audio
                result = await self.process_audio_file(audio_file)
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                return jsonify({'error': str(e)}), 500
                
    async def process_audio_file(self, audio_file) -> Dict[str, Any]:
        """Process uploaded audio file"""
        # Save temporarily
        temp_path = f"/tmp/{audio_file.filename}"
        audio_file.save(temp_path)
        
        try:
            # Load and analyze audio
            logger.info(f"Loading audio: {audio_file.filename}")
            y, sr = librosa.load(temp_path, sr=22050)
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # Get LLM interpretation
            interpretation = await self._get_llm_interpretation(features)
            
            # Update state
            self._update_state(interpretation)
            
            # Emit to connected clients
            self.socketio.emit('state_update', self.current_state)
            
            return {
                'status': 'processed',
                'features': features,
                'interpretation': interpretation,
                'state': self.current_state
            }
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio features"""
        # Basic features
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        
        # Harmonic analysis
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        
        # Energy
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = rms.mean()
        energy_std = rms.std()
        
        return {
            'tempo': float(tempo),
            'spectral_centroid': float(spectral_centroid),
            'chroma_mean': chroma.mean(axis=1).tolist(),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'duration': len(y) / sr
        }
        
    async def _get_llm_interpretation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM interpretation of audio features"""
        prompt = f"""
        Analyze these audio features with minimal, precise language:
        
        {json.dumps(features, indent=2)}
        
        Respond with:
        1. Primary emotion (one word)
        2. Valence (-1 to 1)
        3. Arousal (0 to 1)
        4. Key musical observation
        5. Visual metaphor (brief)
        
        Format: JSON only.
        """
        
        response = await asyncio.to_thread(
            self.llm_client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=200
        )
        
        return json.loads(response.choices[0].message.content)
        
    def _update_state(self, interpretation: Dict[str, Any]):
        """Update current state"""
        self.current_state.update({
            'emotion': interpretation.get('emotion', 'unknown'),
            'valence': interpretation.get('valence', 0.0),
            'arousal': interpretation.get('arousal', 0.0),
            'confidence': interpretation.get('confidence', 0.0),
            'observation': interpretation.get('observation', ''),
            'visual_metaphor': interpretation.get('visual_metaphor', ''),
            'timestamp': datetime.now().isoformat()
        })
        
    def run(self, host='0.0.0.0', port=5000):
        """Start the engine"""
        logger.info(f"Starting Orpheum Dream Engine on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=False)

def main():
    """Main entry point"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'secret_key': os.getenv('SECRET_KEY', 'orpheum-secret')
    }
    
    if not config['openai_api_key']:
        logger.error("OPENAI_API_KEY not found in environment")
        return
    
    engine = OrpheumEngine(config)
    engine.run(
        host=os.getenv('ORPHEUM_HOST', '0.0.0.0'),
        port=int(os.getenv('ORPHEUM_PORT', '5000'))
    )

if __name__ == '__main__':
    main()