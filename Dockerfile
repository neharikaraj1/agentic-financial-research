# ============================================================
# Dockerfile — Agentic Financial Research System
# Multi-stage build for lean production image
# ============================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy & install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env.example .env.example

# Create necessary directories
RUN mkdir -p data/chroma_db data/uploads logs

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
