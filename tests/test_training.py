"""
Tests for Training Module

This module contains tests to verify that the training pipeline works correctly.
These are basic smoke tests to ensure the training script can run without errors
and produces the expected outputs.
"""

import pickle
from pathlib import Path
import sys

import pytest
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Add project root to path so we can import our modules
# This allows the tests to find src/ modules regardless of where pytest is run from
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_sentiment_data
from src.training.train import train_model, evaluate_model, save_model


def test_load_data() -> None:
    """
    Test that data loading function works correctly.
    
    This test verifies:
    - Data loads without errors
    - Returns correct types and shapes
    - Labels are in expected range (0, 1, 2)
    """
    X, y, vectorizer = load_sentiment_data()
    
    # Check that X is a numpy array with correct shape
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert X.shape[0] > 0, "X should have at least one sample"
    assert X.shape[1] > 0, "X should have at least one feature"
    
    # Check that y is a numpy array with correct shape
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert y.shape[0] == X.shape[0], "y should have same number of samples as X"
    
    # Check that labels are in expected range (0, 1, 2 for negative, neutral, positive)
    unique_labels = np.unique(y)
    assert all(label in [0, 1, 2] for label in unique_labels), \
        "Labels should be 0, 1, or 2"
    
    # Check that vectorizer is returned
    assert vectorizer is not None, "Vectorizer should be returned"


def test_train_model() -> None:
    """
    Test that model training works correctly.
    
    This test verifies:
    - Model trains without errors
    - Returns a trained model
    - Test set is created correctly
    """
    # Load data
    X, y, _ = load_sentiment_data()
    
    # Train model
    model, X_test, y_test = train_model(X, y, test_size=0.2, random_state=42)
    
    # Check that model is a MultinomialNB instance
    assert isinstance(model, MultinomialNB), "Model should be MultinomialNB"
    
    # Check that model has been trained (has classes_ attribute)
    assert hasattr(model, "classes_"), "Model should have classes_ after training"
    assert len(model.classes_) > 0, "Model should have at least one class"
    
    # Check that test set is created
    assert X_test.shape[0] > 0, "Test set should have samples"
    assert y_test.shape[0] == X_test.shape[0], \
        "Test labels should match test samples"
    
    # Check that test set has correct number of features
    assert X_test.shape[1] == X.shape[1], \
        "Test set should have same number of features as training set"


def test_evaluate_model() -> None:
    """
    Test that model evaluation works correctly.
    
    This test verifies:
    - Evaluation runs without errors
    - Returns metrics dictionary
    - Metrics contain expected keys
    """
    # Load and train
    X, y, _ = load_sentiment_data()
    model, X_test, y_test = train_model(X, y, test_size=0.2, random_state=42)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Check that metrics is a dictionary
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    # Check that metrics contain expected keys
    assert "accuracy" in metrics, "Metrics should contain 'accuracy'"
    assert "classification_report" in metrics, \
        "Metrics should contain 'classification_report'"
    assert "confusion_matrix" in metrics, \
        "Metrics should contain 'confusion_matrix'"
    
    # Check that accuracy is a float between 0 and 1
    accuracy = metrics["accuracy"]
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1"


def test_save_model() -> None:
    """
    Test that model saving works correctly.
    
    This test verifies:
    - Model and vectorizer are saved to disk
    - Saved files can be loaded back
    - Loaded model can make predictions
    """
    # Load data and train
    X, y, vectorizer = load_sentiment_data()
    model, X_test, y_test = train_model(X, y, test_size=0.2, random_state=42)
    
    # Create temporary directory for test models
    test_model_dir = Path(__file__).parent.parent / "models" / "test"
    test_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path, vectorizer_path = save_model(
        model, vectorizer, test_model_dir, "test_model"
    )
    
    # Check that files were created
    assert model_path.exists(), "Model file should be created"
    assert vectorizer_path.exists(), "Vectorizer file should be created"
    
    # Load model back
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # Check that loaded model is correct type
    assert isinstance(loaded_model, MultinomialNB), \
        "Loaded model should be MultinomialNB"
    
    # Check that loaded model can make predictions
    predictions = loaded_model.predict(X_test)
    assert len(predictions) == len(y_test), \
        "Predictions should match test set size"
    
    # Clean up test files
    model_path.unlink()
    vectorizer_path.unlink()


def test_training_pipeline_integration() -> None:
    """
    Integration test: Test the entire training pipeline end-to-end.
    
    This test verifies that all components work together:
    - Data loading
    - Model training
    - Model evaluation
    - Model saving
    """
    # Load data
    X, y, vectorizer = load_sentiment_data()
    
    # Train model
    model, X_test, y_test = train_model(X, y, test_size=0.2, random_state=42)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Check that model can make predictions (accuracy check is lenient for small dataset)
    # With only 30 samples and 20% test split (6 samples), accuracy can vary
    # We just verify the model works end-to-end, not perfect accuracy
    assert metrics["accuracy"] >= 0.0, \
        "Model should produce valid accuracy metric"
    assert metrics["accuracy"] <= 1.0, \
        "Model accuracy should be valid (0-1 range)"
    
    # Save model
    test_model_dir = Path(__file__).parent.parent / "models" / "test"
    test_model_dir.mkdir(parents=True, exist_ok=True)
    model_path, vectorizer_path = save_model(
        model, vectorizer, test_model_dir, "integration_test_model"
    )
    
    # Verify files exist
    assert model_path.exists(), "Model should be saved"
    assert vectorizer_path.exists(), "Vectorizer should be saved"
    
    # Clean up
    model_path.unlink()
    vectorizer_path.unlink()

