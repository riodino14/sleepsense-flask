# ─────────────────────────────────────────────
# SleepSense Flask API - Dockerfile
# Team CC26-PSU230 | Coding Camp 2026 DBS Foundation
# ─────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="CC26-PSU230"
LABEL description="SleepSense Stress Risk Classifier API"
LABEL version="1.0.0"

# Set working directory
WORKDIR /sleepsense

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project
COPY app/     ./app/
COPY models/  ./models/

# Environment variables (akan di-override oleh .env atau docker run -e)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port
EXPOSE 5000

# Jalankan dengan Gunicorn (production-grade WSGI server)
# 2 workers, timeout 120s (model loading butuh waktu)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.main:app"]
