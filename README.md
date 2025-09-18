# SYNAPSE

**S**calable **Y**outh **N**euroimaging **A**nalysis **P**ipeline for **S**cientific **E**xploration

A reproducible, config-driven neuroimaging pipeline for ABCD dataset anxiety analysis.

## Quick Start

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

### Docker Development
```bash
# Build and run with Docker Compose
docker-compose up

# Access Jupyter at http://localhost:8888
```

## Project Structure

```
SYNAPSE/
├── src/synapse/          # Python package
├── notebooks/            # Jupyter notebooks
├── configs/              # Configuration files
├── tests/                # Test suite
├── data/                 # Data directory
└── docs/                 # Documentation
```
