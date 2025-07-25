// Orpheum Dream Engine - Main Application

class Orpheum {
    constructor() {
        this.socket = null;
        this.visualizer = null;
        this.state = {
            emotion: 'neutral',
            valence: 0,
            arousal: 0,
            processing: false
        };
        
        this.init();
    }
    
    init() {
        // Initialize socket connection
        this.socket = io();
        
        // Initialize visualizer
        this.visualizer = new OrpheumVisualizer('dream-canvas');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Create neural grid
        this.createNeuralGrid();
        
        console.log('Orpheum initialized');
    }
    
    setupEventListeners() {
        // Socket events
        this.socket.on('connect', () => {
            document.getElementById('status').textContent = 'CONNECTED';
            document.getElementById('status').classList.add('active');
        });
        
        this.socket.on('state_update', (state) => {
            this.updateState(state);
        });
        
        // File upload
        document.getElementById('audio-input').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });
    }
    
    updateState(state) {
        this.state = state;
        
        // Update UI
        document.getElementById('emotion').textContent = state.emotion || '—';
        document.getElementById('valence').textContent = 
            state.valence ? state.valence.toFixed(2) : '0.00';
        document.getElementById('arousal').textContent = 
            state.arousal ? state.arousal.toFixed(2) : '0.00';
        document.getElementById('node').textContent = state.active_node || 'IDLE';
        
        if (state.observation) {
            document.getElementById('observation').textContent = state.observation;
        }
        
        // Update visualizer
        this.visualizer.updateParams(state);
        
        // Update neural grid
        this.updateNeuralActivity();
    }
    
    async handleFileUpload(file) {
        if (!file) return;
        
        document.getElementById('status').textContent = 'PROCESSING';
        
        const formData = new FormData();
        formData.append('audio', file);
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            console.log('Processing result:', result);
            
            document.getElementById('status').textContent = 'DREAMING';
            
        } catch (error) {
            console.error('Upload error:', error);
            document.getElementById('status').textContent = 'ERROR';
        }
    }
    
    createNeuralGrid() {
        const grid = document.getElementById('neural-grid');
        for (let i = 0; i < 200; i++) {
            const cell = document.createElement('div');
            cell.className = 'neural-cell';
            grid.appendChild(cell);
        }
    }
    
    updateNeuralActivity() {
        const cells = document.querySelectorAll('.neural-cell');
        cells.forEach(cell => {
            if (Math.random() > 0.95) {
                cell.classList.add('active');
                setTimeout(() => cell.classList.remove('active'), 200);
            }
        });
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    window.orpheum = new Orpheum();
});