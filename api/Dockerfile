# CycloneX Backend Dockerfile
# Optimized for local containers, Docker Compose, and Hugging Face Spaces (Docker SDK)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install minimal OS dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy configs, models, and API source code
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY data/ /app/data/
COPY api/ /app/api/

# Create a non-root user (standard practice for Hugging Face Spaces & security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

# Launch FastAPI via Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
