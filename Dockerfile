# ──────────────────────────────────────────────────
# MisinformationModerationEnv — Dockerfile
# HuggingFace Spaces compatible (port 7860)
# ──────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for HF Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py tasks.py client.py __init__.py ./
COPY server/ ./server/

# Ensure server/__init__.py exists
RUN touch server/__init__.py

# Metadata
LABEL org.opencontainers.image.title="MisinformationModerationEnv" \
    org.opencontainers.image.description="OpenEnv RL environment for content moderation" \
    org.opencontainers.image.version="1.0.0"

# Environment variables with sane defaults
ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
