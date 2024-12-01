FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PADDLE_HOME=/tmp/paddleocr

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Create temporary directories with proper permissions
RUN mkdir -p /tmp/matplotlib /tmp/ultralytics /tmp/paddleocr /app/.cache && \
    chmod -R 777 /tmp/matplotlib /tmp/ultralytics /tmp/paddleocr /app/.cache && \
    chown -R 1000:1000 /tmp/matplotlib /tmp/ultralytics /tmp/paddleocr /app/.cache

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download models with specific cache locations
RUN PYTHONPATH=/app \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    python3 -c "import os; os.environ['PADDLE_HOME']='/tmp/paddleocr'; from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# Add a non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]