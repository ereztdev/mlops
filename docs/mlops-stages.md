# MLOps Stages

This document outlines the different stages of the MLOps lifecycle as implemented in this project.

## Stage 1: Training

**Status**: ✅ Implemented

The training stage involves:
- Data loading and preprocessing (`src/data/load_data.py`)
- Model training (`src/training/train.py`)
- Model evaluation (accuracy, classification report, confusion matrix)
- Saving model artifacts (model.pkl and vectorizer.pkl to `models/` directory)

**Implementation Details:**
- Uses scikit-learn's Multinomial Naive Bayes classifier for sentiment analysis
- TF-IDF vectorization for text preprocessing
- Train/test split (80/20) for model evaluation
- Model and vectorizer saved as pickle files for inference
- Comprehensive evaluation metrics printed during training

**Usage:**
```bash
# Run training
python src/training/train.py

# Or in Docker
docker run --rm -v $(pwd):/app mlops:latest python src/training/train.py
```

## Stage 2: Automated CI Pipeline

**Status**: ✅ Implemented

The CI pipeline automates quality checks and training on every code change:

**GitHub Actions Workflow** (`.github/workflows/ci.yml`):
- **Lint and Test Job**: 
  - Runs on every push and pull request
  - Code linting with flake8 (style and error checking)
  - Runs all tests with pytest
  - Code coverage reporting
- **Training Job**:
  - Trains the model to ensure pipeline works
  - Validates training script executes successfully
  - Uploads model artifacts for download

**Benefits:**
- Automatic validation on every code change
- Catches errors before they reach production
- Ensures training pipeline remains functional
- Model artifacts stored for 7 days

**Usage:**
- Automatically runs on push to `main` branch
- Automatically runs on pull requests
- View results in GitHub Actions tab
- Download artifacts from workflow runs

## Stage 3: Artifact Handling

**Status**: ✅ Implemented

Artifact handling includes:
- **Model serialization**: Models saved as `.pkl` files (pickle format)
- **Metrics storage**: Evaluation metrics saved as `metrics.json` (JSON format)
- **CI/CD artifact persistence**: Artifacts uploaded to GitHub Actions and stored for 7 days
- **Artifact structure**:
  - `models/sentiment_model.pkl` - Trained classifier
  - `models/sentiment_model_vectorizer.pkl` - Fitted text vectorizer
  - `models/metrics.json` - Evaluation metrics (accuracy, classification report, confusion matrix)

**Implementation Details:**
- Metrics include accuracy, precision, recall, F1-score per class
- JSON format allows easy parsing by other tools
- CI pipeline automatically uploads all artifacts after training
- Artifacts can be downloaded from GitHub Actions workflow runs

**Usage:**
- Artifacts are automatically created during training
- CI/CD automatically uploads artifacts for each training run
- Download artifacts from GitHub Actions workflow run page

## Stage 4: Model Registry

**Status**: ✅ Implemented

The model registry provides:
- **Model versioning**: Models registered in MLflow with automatic versioning
- **Experiment tracking**: All training runs tracked with parameters and metrics
- **Model metadata**: Training parameters, dataset info, and configuration stored
- **Artifact storage**: Models, vectorizers, and metrics stored in MLflow

**Implementation Details:**
- Uses MLflow for experiment tracking and model registry
- Local tracking URI: `mlruns/` directory
- Experiment: `sentiment-analysis`
- Registered model: `SentimentAnalysisModel`
- Each run logs:
  - Parameters: dataset_size, num_features, model_type, test_size, etc.
  - Metrics: accuracy, per-class precision/recall/F1-score
  - Artifacts: model, vectorizer, metrics.json, confusion_matrix.json

**Usage:**
```bash
# Run training (automatically logs to MLflow)
python src/training/train.py

# View MLflow UI
mlflow ui --backend-store-uri mlruns
# Then open http://localhost:5000 in browser
```

**Benefits:**
- Track all training experiments in one place
- Compare model performance across runs
- Reproduce any training run with logged parameters
- Version control for models
- Easy model retrieval for deployment

**CI/CD Integration:**
- MLflow tracking data (`mlruns/`) uploaded as artifact in CI pipeline
- Preserves experiment history for dashboard visualization (see Stage 7)

## Stage 5: Inference

**Status**: ✅ Implemented

Inference capabilities:
- **REST API endpoints** (`src/inference/app.py`):
  - `POST /predict` - Single text prediction
  - `POST /predict/batch` - Batch predictions
  - `GET /health` - Health check
  - `GET /` - Interactive web UI
  - `GET /docs` - Automatic API documentation (Swagger UI)
- **Model loading** (`src/inference/load_model.py`):
  - Loads trained model and vectorizer from disk
  - Model cached in memory for fast predictions
  - Error handling for missing model files
- **Request processing**:
  - Pydantic models for request/response validation
  - Text preprocessing using trained vectorizer
  - Sentiment prediction with confidence scores
- **Response generation**:
  - JSON responses with sentiment and confidence
  - Web UI for interactive testing

**Implementation Details:**
- FastAPI framework for modern, fast API
- Automatic OpenAPI/Swagger documentation
- Model loaded once at startup (cached in memory)
- Web UI included for easy testing and demos
- Supports both single and batch predictions

**Usage:**
```bash
# Start the API server
python src/inference/app.py

# Or in Docker
docker run -p 8000:8000 -v $(pwd):/app mlops:latest python src/inference/app.py

# Then access:
# - Web UI: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

## Stage 6: Deployment

**Status**: ✅ Implemented (Staging + Production)

**Staging Deployment:**
- **CI/CD Integration**: Automatic deployment job in GitHub Actions
- **Docker Image**: Builds and pushes inference API to GitHub Container Registry (ghcr.io)
- **Image Tags**: 
  - `latest` - Latest version from main branch
  - `main-<sha>` - Tagged with commit SHA for traceability
- **Deployment Job**: Runs after successful lint/test and training
- **Staging Environment**: Configured in GitHub Actions (placeholder URL)

**Implementation Details:**
- Uses Docker Buildx for efficient image builds
- Pushes to `ghcr.io/ereztdev/mlops` (GitHub Container Registry)
- Model artifacts downloaded from training job
- Image cached for faster subsequent builds
- Deployment step is a placeholder (ready for actual infrastructure)

**Usage:**
```bash
# Pull and run the deployed image
docker pull ghcr.io/ereztdev/mlops:latest
docker run -p 8000:8000 ghcr.io/ereztdev/mlops:latest python src/inference/app.py
```

**Production Deployment:**
- **Status**: ✅ Implemented (Commit 9)
- **Manual Approval Gate**: Production deployment requires manual approval in GitHub Actions
- **Environment Promotion**: Automatic promotion from staging to production after approval
- **Production Image Tags**: 
  - `production` - Latest production version
  - `production-<sha>` - Tagged with commit SHA
  - `prod-<sha>` - Alternative tag format
- **Deployment Workflow**:
  1. Code pushed to main → Staging deploys automatically
  2. Staging deployment succeeds → Production job waits for approval
  3. Manual approval in GitHub Actions UI → Production deploys
  4. Production deployment completes → Service available

**How to Approve Production Deployment:**
1. Go to GitHub Actions tab in repository
2. Find the workflow run that completed staging
3. Click on "Deploy to Production" job
4. Click "Review deployments" button
5. Approve the deployment
6. Production deployment will proceed automatically

**Rollback Capabilities:**
- Previous production images remain in registry
- Can redeploy previous version by pulling older image tag
- When expanding to a more robust production grade, add automated rollback on health check failures

## Stage 7: Monitoring

**Status**: ✅ Partially Implemented (Scaffold)

**Note:** Dashboard/visualization requirements added in Commit 6 (MLflow Integration)

**Implemented Components:**
- **Prediction Logging** (`src/monitoring/collector.py`):
  - Logs all predictions to `logs/predictions.jsonl` (JSON Lines format)
  - Tracks: text, prediction, confidence, timestamp, model version
  - Integrated into inference API (automatic logging on every prediction)
- **Prediction Statistics**:
  - Get recent predictions and statistics via collector
  - Health endpoint includes monitoring stats
  - Tracks: total predictions, sentiment distribution, confidence metrics
- **Drift Detection Scaffold**:
  - Placeholder function for drift detection
  - Simple confidence-based detection (placeholder)
  - Ready for production-grade implementation

**Implementation Details:**
- `PredictionCollector` class for logging and statistics
- Automatic logging in `/predict` and `/predict/batch` endpoints
- Logs stored in `logs/predictions.jsonl` (JSON Lines format)
- Health endpoint (`/health`) includes monitoring statistics
- Drift detection function ready for enhancement

**Usage:**
```python
from src.monitoring.collector import PredictionCollector, detect_drift

# Get statistics
collector = PredictionCollector()
stats = collector.get_prediction_stats()
print(stats)

# Get recent predictions
recent = collector.get_recent_predictions(limit=100)

# Detect drift (placeholder)
drift_result = detect_drift(recent)
```

**Frontend Dashboard** (planned for future):
- Experiment history visualization (all training runs)
- Model performance metrics over time
- Model comparison (accuracy, precision, recall across versions)
- Training parameters visualization
- Model version timeline
- Integration with MLflow tracking data
- Real-time prediction monitoring

**Implementation Options for Dashboard:**
- Option A: Use MLflow's built-in UI (`mlflow ui`) - quick start
- Option B: Build custom dashboard (React/Vue + FastAPI backend) - full control
- Option C: Integrate MLflow UI into web interface - hybrid approach

