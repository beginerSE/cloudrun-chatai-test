# Multi-stage Dockerfile for GCP Cloud Run
# Stage 1: Build stage
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production stage
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only (PostgreSQL client library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py .
COPY static/ static/
COPY templates/ templates/

# Create directory for database migrations (if needed)
RUN mkdir -p database

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV FLASK_ENV=production

# Cloud Run expects the app to listen on $PORT
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 300 --worker-class gthread app:app
