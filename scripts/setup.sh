#!/bin/bash
# Orpheum Dream Engine Setup

echo "Setting up Orpheum Dream Engine..."

# Create directories
mkdir -p uploads outputs workflows logs

# Install dependencies
pip install -r requirements.txt

# Download Three.js
mkdir -p web/static/lib
curl -o web/static/lib/three.min.js https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
curl -o web/static/lib/socket.io.min.js https://cdn.socket.io/4.5.4/socket.io.min.js

echo "✓ Setup complete"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Add your OpenAI API key"
echo "3. Run: python -m orpheum.engine"