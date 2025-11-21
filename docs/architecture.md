# Architecture Overview

This document describes the high-level architecture of the MLOps project.

## System Components

### 1. Training Pipeline

The training pipeline is responsible for:
- Loading and preprocessing data
- Training machine learning models
- Evaluating model performance
- Saving trained models and metrics

**Location**: `src/training/`

### 1.1 Training Container

The training pipeline is containerized using Docker for consistent execution environments:

- **Base Image**: Python 3.10-slim
- **Dependencies**: Installed from `requirements.txt`
- **Working Directory**: `/app` (project root)
- **Execution**: Training script runs inside container

**Usage:**
```bash
# Build the image
docker build -t mlops:latest .

# Run training in container
docker run --rm -v $(pwd):/app mlops:latest python src/training/train.py
```

The containerization ensures:
- Consistent Python and dependency versions across environments
- Isolation from host system dependencies
- Reproducible training runs
- Easy deployment to different environments

### 2. Model Registry

Models are versioned and tracked using MLflow, which provides:
- Model versioning
- Experiment tracking
- Model metadata storage
- Model artifact storage

**Location**: MLflow server (local or remote)

### 3. Inference API

The inference API serves trained models via a REST API:
- FastAPI-based REST endpoints
- Model loading and caching
- Request validation
- Response formatting

**Location**: `src/inference/`

### 4. CI/CD Pipeline

GitHub Actions workflows automate:
- Code linting and testing
- Model training on code changes
- Model artifact storage
- Deployment to staging/production

**Location**: `.github/workflows/`

### 5. Containerization

Docker containers provide:
- Consistent execution environments
- Isolation from host system
- Easy deployment
- Reproducible builds

**Location**: `Dockerfile`, `docker-compose.yml` (future)

## Data Flow

```
Data → Training Script → Model Artifact → MLflow Registry → Inference API → Predictions
```

## Technology Stack

- **Language**: Python 3.10
- **ML Framework**: scikit-learn
- **Model Registry**: MLflow
- **API Framework**: FastAPI
- **CI/CD**: GitHub Actions
- **Containerization**: Docker

