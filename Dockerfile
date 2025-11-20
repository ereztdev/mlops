# Use Python 3.10 as base image
# This is a stable version widely supported in the ML ecosystem
FROM python:3.10-slim

# Set working directory inside the container
# All commands will run from this directory
WORKDIR /app

# Set environment variables
# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements file first
# This allows Docker to cache the dependency installation layer
# If requirements.txt doesn't change, Docker won't reinstall dependencies
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not storing pip cache
# --upgrade ensures we get the latest compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This is done after installing dependencies for better caching
COPY . .

# Make scripts executable (ignore errors if scripts don't exist yet)
# This allows health check and other utility scripts to be run directly
RUN chmod +x scripts/*.py scripts/*.sh 2>/dev/null || true

# Default command (can be overridden when running the container)
# This allows the container to run different scripts
CMD ["python", "--version"]

