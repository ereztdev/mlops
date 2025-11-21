"""
Model Loading Module

This module provides functions to load trained models and vectorizers
for making predictions on new text data.
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def load_model_and_vectorizer(
    model_dir: Optional[Path] = None,
    model_name: str = "sentiment_model"
) -> Tuple[MultinomialNB, TfidfVectorizer]:
    """
    Load trained model and vectorizer from disk.
    
    This function loads the saved model and vectorizer files that were
    created during training. Both are needed to make predictions:
    - Vectorizer: converts new text to the same feature format used in training
    - Model: makes the actual sentiment prediction
    
    Args:
        model_dir: Directory containing model files. If None, uses default models/ directory
        model_name: Base name for model files (without extension)
    
    Returns:
        Tuple containing:
            - model: Loaded MultinomialNB classifier
            - vectorizer: Loaded TfidfVectorizer
    
    Raises:
        FileNotFoundError: If model or vectorizer files don't exist
    """
    # Default to models/ directory in project root
    if model_dir is None:
        # Get project root (3 levels up from this file: src/inference/load_model.py)
        project_root = Path(__file__).parent.parent.parent
        model_dir = project_root / "models"
    
    # Define file paths
    model_path = model_dir / f"{model_name}.pkl"
    vectorizer_path = model_dir / f"{model_name}_vectorizer.pkl"
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train a model first using: python src/training/train.py"
        )
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}\n"
            f"Please train a model first using: python src/training/train.py"
        )
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_sentiment(
    text: str,
    model: MultinomialNB,
    vectorizer: TfidfVectorizer
) -> Tuple[str, float]:
    """
    Predict sentiment for a single text input.
    
    This function takes raw text, preprocesses it using the vectorizer,
    and makes a prediction using the trained model.
    
    Args:
        text: Input text to classify
        model: Trained MultinomialNB classifier
        vectorizer: Fitted TfidfVectorizer for text preprocessing
    
    Returns:
        Tuple containing:
            - sentiment: Predicted sentiment label ("negative", "neutral", or "positive")
            - confidence: Prediction confidence score (probability)
    """
    # Convert text to feature vector using the same vectorizer from training
    # The vectorizer must be the same one used during training
    text_vector = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    
    # Get prediction probabilities for confidence score
    probabilities = model.predict_proba(text_vector)[0]
    confidence = float(probabilities[prediction])
    
    # Map numeric label to sentiment name
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = label_map[prediction]
    
    return sentiment, confidence

