FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Install package in development mode with dev dependencies
RUN pip install -e ".[dev]"

# Copy pre-commit config
COPY .pre-commit-config.yaml .

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
