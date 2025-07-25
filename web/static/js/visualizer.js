// Orpheum Visualizer - Minimal mysterious visualization

class OrpheumVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.monolith = null;
        this.particles = [];
        
        this.params = {
            emotion: 'neutral',
            valence: 0,
            arousal: 0
        };
        
        this.init();
        this.animate();
    }
    
    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000000, 0.002);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.z = 50;
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000, 0);
        
        // Create monolith
        this.createMonolith();
        
        // Create minimal particles
        this.createParticles();
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
        this.scene.add(ambientLight);
        
        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }
    
    createMonolith() {
        const geometry = new THREE.BoxGeometry(1, 20, 1);
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            wireframe: true,
            opacity: 0.3,
            transparent: true
        });
        
        this.monolith = new THREE.Mesh(geometry, material);
        this.scene.add(this.monolith);
    }
    
    createParticles() {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        
        for (let i = 0; i < 100; i++) {
            positions.push(
                (Math.random() - 0.5) * 100,
                (Math.random() - 0.5) * 100,
                (Math.random() - 0.5) * 100
            );
        }
        
        geometry.setAttribute('position', 
            new THREE.Float32BufferAttribute(positions, 3));
        
        const material = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.5,
            opacity: 0.3,
            transparent: true
        });
        
        const particles = new THREE.Points(geometry, material);
        this.scene.add(particles);
        this.particles.push(particles);
    }
    
    updateParams(state) {
        this.params = {
            emotion: state.emotion || 'neutral',
            valence: state.valence || 0,
            arousal: state.arousal || 0
        };
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Subtle monolith rotation
        if (this.monolith) {
            this.monolith.rotation.y += 0.001 + this.params.arousal * 0.002;
            
            // Scale based on valence
            const scale = 1 + this.params.valence * 0.2;
            this.monolith.scale.y = scale;
        }
        
        // Particle movement
        this.particles.forEach(particleSystem => {
            particleSystem.rotation.y += 0.0005;
        });
        
        this.renderer.render(this.scene, this.camera);
    }
    
    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}