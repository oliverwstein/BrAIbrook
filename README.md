# BrAIbrook
# README.md
# Manuscript Transcriber

An iterative transcription system combining DeepSeek's vision capabilities with Kraken's specialized HTR for historical documents.

## Installation

```bash
# Create and activate virtual environment
python3 -m venv kraken_env
source kraken_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code for the transcription system
  - `input_processing/`: Document analysis and preprocessing
  - `transcription/`: Kraken HTR integration and processing
  - `verification/`: Content review and refinement
  - `output/`: Text generation and metadata handling
- `tests/`: Unit tests and integration tests
- `models/`: Trained models and model configurations
- `data/`: Input and output data
- `docs/`: Documentation and guides

## Usage

[Usage instructions will go here]

## Development

[Development instructions will go here]

---

# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
kraken_env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/raw/*
data/processed/*
models/kraken/*
models/deepseek/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!models/kraken/.gitkeep
!models/deepseek/.gitkeep

# OS specific
.DS_Store
.AppleDouble
.LSOverride