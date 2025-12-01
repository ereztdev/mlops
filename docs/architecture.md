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
- **Model versioning**: Each trained model is registered with a version number
- **Experiment tracking**: All training runs are logged with parameters and metrics
- **Model metadata storage**: Training parameters, dataset info, and model configuration
- **Model artifact storage**: Models and related artifacts stored in MLflow

**Implementation:**
- MLflow tracking URI: `mlruns/` directory (local storage)
- Experiment name: `sentiment-analysis`
- Registered model: `SentimentAnalysisModel`
- Each training run logs:
  - Parameters: dataset size, model type, test size, random state
  - Metrics: accuracy, precision, recall, F1-score per class
  - Artifacts: model, vectorizer, metrics JSON, confusion matrix

**Location**: `mlruns/` directory (local) or remote MLflow server

**Usage:**
```bash
# View MLflow UI to see experiments and runs
mlflow ui --backend-store-uri mlruns

# Access at http://localhost:5000
```

**When expanding to a more robust production grade:**
- Deploy MLflow server (remote tracking)
- Use MLflow Model Registry for model promotion workflow
- Integrate with CI/CD for automatic model registration

### 3. Inference API

The inference API serves trained models via a REST API:
- FastAPI-based REST endpoints
- Model loading and caching
- Request validation
- Response formatting

**Location**: `src/inference/`

**Endpoints:**
- `GET /` - Interactive web UI for sentiment analysis
- `POST /predict` - Single text prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check endpoint
- `GET /docs` - Automatic API documentation (Swagger UI)

**Features:**
- Model loaded once at startup (cached in memory for performance)
- Pydantic models for request/response validation
- Automatic OpenAPI documentation
- Web UI included for easy testing
- Error handling for missing models or prediction failures

**Usage:**
```bash
# Start API server
python src/inference/app.py

# Access web UI: http://localhost:8000
# Access API docs: http://localhost:8000/docs
```

### 4. CI/CD Pipeline

GitHub Actions workflows automate:
- Code linting and testing
- Model training on code changes
- Model artifact storage
- Deployment to staging/production

**Location**: `.github/workflows/`

**Workflow Jobs:**
1. **Lint and Test**: Code quality checks and test execution
2. **Train Model**: Model training and artifact generation
3. **Deploy to Staging**: Docker image build and push to GitHub Container Registry (automatic)
4. **Deploy to Production**: Production deployment with manual approval gate

**Deployment:**
- **Container Registry**: GitHub Container Registry (ghcr.io)
- **Image Repository**: `ghcr.io/ereztdev/mlops`
- **Staging Tags**: `latest` and `main-<sha>` for versioning
- **Production Tags**: `production`, `production-<sha>`, `prod-<sha>`
- **Staging Environment**: Automatic deployment after successful training
- **Production Environment**: Manual approval required before deployment
- **Promotion Workflow**: Staging → (manual approval) → Production

### 5. Containerization

Docker containers provide:
- Consistent execution environments
- Isolation from host system
- Easy deployment
- Reproducible builds

**Location**: `Dockerfile`, `docker-compose.yml` (future)

### 6. Monitoring

Monitoring system tracks model performance and detects data drift:

- **Prediction Logging**: All predictions logged to `logs/predictions.jsonl`
- **Statistics Collection**: Tracks prediction counts, sentiment distribution, confidence metrics
- **Drift Detection**: Placeholder for detecting data/model drift (ready for production implementation)
- **Health Monitoring**: Health endpoint includes monitoring statistics

**Location**: `src/monitoring/`

**Integration:**
- Automatically logs all predictions from inference API
- Statistics available via health endpoint
- Logs stored in JSON Lines format for easy analysis

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

