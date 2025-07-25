from setuptools import setup, find_packages

setup(
    name="orpheum-dream-engine",
    version="1.0.0",
    description="Audio-to-visual dream generator powered by LLMs",
    author="Orpheum Collective",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "flask>=3.0.0",
        "flask-socketio>=5.3.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "orpheum=orpheum.engine:main",
        ],
    },
)