# ==============================================================================
# LungCare AI Backend — Production Dockerfile (Render Deployment)
# ==============================================================================
FROM python:3.11-slim

# Runtime & TensorFlow environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    ENVIRONMENT=production \
    DEBUG=false \
    ENABLE_DOCS=false \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Minimal system library required for OpenMP / CPU runtime compatibility
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install web and ML dependencies separately to leverage layer caching
COPY backend/requirements.txt backend/requirements-ml.txt /app/backend/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt -r /app/backend/requirements-ml.txt

# Copy backend source and all four required DenseNet121 model artifacts
COPY backend/app /app/backend/app
COPY models/LungCare_AI_Model /app/models/LungCare_AI_Model

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/backend

EXPOSE 8000

# Bind to 0.0.0.0 and dynamically use Render's $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
