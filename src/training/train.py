"""
Training Script

This script trains a sentiment analysis model using scikit-learn.
It loads data, trains a classifier, evaluates performance, and saves
the trained model for later use in inference.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to Python path so we can import src modules
# This allows the script to run from any directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_sentiment_data, get_data_info


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MultinomialNB, np.ndarray, np.ndarray]:
    """
    Train a sentiment analysis model.
    
    This function splits the data into training and testing sets,
    trains a Multinomial Naive Bayes classifier, and returns the
    trained model along with test data for evaluation.
    
    Args:
        X: Feature matrix (vectorized text data)
        y: Target labels (0=negative, 1=neutral, 2=positive)
        test_size: Proportion of data to use for testing (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple containing:
            - model: Trained MultinomialNB classifier
            - X_test: Test feature matrix
            - y_test: Test labels
    """
    # Split data into training and testing sets
    # This is a fundamental ML practice: we train on one set and
    # evaluate on a separate set to check if the model generalizes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensures each class is represented proportionally
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Create and train the model
    # Multinomial Naive Bayes is well-suited for text classification
    # It's simple, fast, and works well with TF-IDF features
    model = MultinomialNB()
    
    # Train the model on the training data
    # This is where the model learns the patterns in the data
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model, X_test, y_test


def evaluate_model(
    model: MultinomialNB,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate the trained model on test data.
    
    This function makes predictions and calculates various metrics
    to assess model performance.
    
    Args:
        model: Trained classifier
        X_test: Test feature matrix
        y_test: True test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy (overall correctness)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate detailed classification report
    # This shows precision, recall, and F1-score for each class
    report = classification_report(
        y_test, y_pred,
        target_names=["negative", "neutral", "positive"],
        output_dict=True
    )
    
    # Generate confusion matrix
    # Shows how many predictions were correct/incorrect for each class
    cm = confusion_matrix(y_test, y_pred)
    
    # Print evaluation results
    print("\n" + "=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["negative", "neutral", "positive"]
    ))
    
    print("\nConfusion Matrix:")
    print("Rows = True labels, Columns = Predicted labels")
    print("        neg  neu  pos")
    label_names = ["neg", "neu", "pos"]
    for i, row in enumerate(cm):
        print(f"{label_names[i]:4}  {row}")
    
    # Return metrics as dictionary for potential logging/saving
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


def save_model(
    model: MultinomialNB,
    vectorizer,
    model_dir: Path,
    model_name: str = "sentiment_model"
) -> Tuple[Path, Path]:
    """
    Save the trained model and vectorizer to disk.
    
    The model and vectorizer are saved as pickle files so they can
    be loaded later for making predictions.
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TfidfVectorizer used for preprocessing
        model_dir: Directory where models will be saved
        model_name: Base name for the model files
    
    Returns:
        Tuple containing paths to saved model and vectorizer files
    """
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    model_path = model_dir / f"{model_name}.pkl"
    vectorizer_path = model_dir / f"{model_name}_vectorizer.pkl"
    
    # Save the model
    # pickle is Python's standard way to serialize objects
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save the vectorizer (needed to preprocess new text for inference)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    return model_path, vectorizer_path


def save_metrics(
    metrics: dict,
    model_dir: Path,
    metrics_name: str = "metrics"
) -> Path:
    """
    Save evaluation metrics to a JSON file.
    
    This allows metrics to be easily read by other tools, CI/CD pipelines,
    or for tracking model performance over time.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        model_dir: Directory where metrics will be saved
        metrics_name: Base name for the metrics file
    
    Returns:
        Path to the saved metrics file
    """
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file path
    metrics_path = model_dir / f"{metrics_name}.json"
    
    # Convert metrics to JSON-serializable format
    # Some values might be numpy types that need conversion
    json_metrics = {
        "accuracy": float(metrics["accuracy"]),
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"]
    }
    
    # Save metrics as JSON
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics_path


def main() -> None:
    """
    Main training pipeline.
    
    This function orchestrates the entire training process:
    1. Load and preprocess data
    2. Train the model
    3. Evaluate performance
    4. Save the trained artifacts
    """
    print("=" * 50)
    print("Sentiment Analysis Model Training")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    X, y, vectorizer = load_sentiment_data()
    get_data_info(X, y)
    
    # Step 2: Train model
    print("\n[Step 2] Training model...")
    model, X_test, y_test = train_model(X, y)
    
    # Step 3: Evaluate model
    print("\n[Step 3] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 4: Save model and metrics
    print("\n[Step 4] Saving artifacts...")
    # Create models directory in project root
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models"
    model_path, vectorizer_path = save_model(model, vectorizer, model_dir)
    metrics_path = save_metrics(metrics, model_dir)
    
    print("\n" + "=" * 50)
    print("Training pipeline complete!")
    print("=" * 50)
    print(f"\nModel artifacts saved:")
    print(f"  - Model: {model_path}")
    print(f"  - Vectorizer: {vectorizer_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"\nModel accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")


if __name__ == "__main__":
    main()

