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

**Status**: To be implemented

The CI pipeline will:
- Run tests on code changes
- Execute training jobs
- Validate model performance
- Store artifacts

## Stage 3: Artifact Handling

**Status**: To be implemented

Artifact handling includes:
- Model serialization
- Metrics storage
- Version tracking
- Artifact persistence

## Stage 4: Model Registry

**Status**: To be implemented

The model registry provides:
- Model versioning
- Experiment tracking
- Model metadata
- Artifact storage

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

Monitoring includes:
- Model performance tracking
- Data drift detection
- Prediction logging
- Alerting

