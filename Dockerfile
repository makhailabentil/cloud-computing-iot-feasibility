FROM python:3.11-slim

# Install system dependencies including Tesseract OCR and LaTeX
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    texlive-latex-base \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY lucidpie/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads, outputs, and data (inside lucidpie)
RUN mkdir -p lucidpie/uploads lucidpie/outputs lucidpie/data && \
    chmod 755 lucidpie/uploads lucidpie/outputs lucidpie/data

# Set environment variables
ENV PRODUCTION=true
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
# Using uvicorn directly for now (gunicorn had issues with root route)
# For production with multiple workers, use: gunicorn lucidpie.web_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
CMD ["python", "start_web_server.py"]
