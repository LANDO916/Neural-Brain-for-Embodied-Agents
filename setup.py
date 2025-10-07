#!/usr/bin/env python3
"""
Setup script for the Multi-Dimensional Audio Analysis Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="audio-analysis-agent",
    version="1.0.0",
    author="Audio Analysis Agent Team",
    author_email="contact@audioanalysis.ai",
    description="A comprehensive free audio analysis system implementing parallel deconstruction across three dimensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-analysis-agent",
    packages=find_packages(),
    py_modules=["audio_analysis_agent", "demo"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "notebook>=6.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "audio-analysis=audio_analysis_agent:main",
            "audio-analysis-demo=demo:run_demo",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "audio", "music", "analysis", "machine-learning", "signal-processing",
        "speech-recognition", "music-information-retrieval", "librosa",
        "free", "open-source", "parallel-processing", "sentiment-analysis"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/audio-analysis-agent/issues",
        "Source": "https://github.com/yourusername/audio-analysis-agent",
        "Documentation": "https://github.com/yourusername/audio-analysis-agent/blob/main/README.md",
    },
)