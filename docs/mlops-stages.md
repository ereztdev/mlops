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

**Status**: To be implemented

Inference capabilities:
- REST API endpoints
- Model loading
- Request processing
- Response generation

## Stage 6: Deployment

**Status**: To be implemented

Deployment stages:
- Staging environment
- Production environment
- Environment promotion
- Rollback capabilities

## Stage 7: Monitoring

**Status**: To be implemented

**Note:** Dashboard/visualization requirements added in Commit 6 (MLflow Integration)

Monitoring includes:
- Model performance tracking
- Data drift detection
- Prediction logging
- Alerting
- **Frontend Dashboard** (planned):
  - Experiment history visualization (all training runs)
  - Model performance metrics over time
  - Model comparison (accuracy, precision, recall across versions)
  - Training parameters visualization
  - Model version timeline
  - Integration with MLflow tracking data

**Implementation Options:**
- Option A: Use MLflow's built-in UI (`mlflow ui`) - quick start
- Option B: Build custom dashboard (React/Vue + FastAPI backend) - full control
- Option C: Integrate MLflow UI into web interface - hybrid approach

