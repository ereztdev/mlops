"""
Data Loading Module

This module provides functions to load and preprocess sentiment analysis data.
For this learning project, we'll use a simple synthetic dataset to focus on
MLOps practices rather than data collection complexity.
"""

from typing import Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_sentiment_data() -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Load and preprocess sentiment analysis training data.
    
    This function creates a simple synthetic dataset for sentiment analysis.
    When expanding to a more robust production grade, this would load from
    files, databases, or APIs.
    
    Returns:
        Tuple containing:
            - X: Feature matrix (vectorized text data)
            - y: Target labels (0=negative, 1=neutral, 2=positive)
            - vectorizer: Fitted TfidfVectorizer for later use in inference
    
    Example:
        >>> X, y, vectorizer = load_sentiment_data()
        >>> print(X.shape)  # (number of samples, number of features)
        >>> print(y.shape)  # (number of samples,)
    """
    # Sample text data with sentiment labels
    # When expanding to a more robust production grade, this would come from
    # a dataset file, database, or API. For now, we use simple examples to
    # establish the MLOps workflow foundation.
    texts = [
        # Negative sentiment (label: 0)
        "I hate this product, it's terrible",
        "This is awful and disappointing",
        "Very bad experience, not recommended",
        "Poor quality, waste of money",
        "Disappointed with the service",
        "This movie was boring and slow",
        "Not worth the price at all",
        "Terrible customer service",
        "I regret buying this",
        "Worst purchase I've ever made",
        
        # Neutral sentiment (label: 1)
        "The product arrived on time",
        "It does what it's supposed to do",
        "Average quality, nothing special",
        "It's okay, nothing more",
        "Standard features, as expected",
        "The item is fine",
        "Meets basic requirements",
        "Nothing to complain about",
        "It works as described",
        "Basic functionality is present",
        
        # Positive sentiment (label: 2)
        "I love this product, it's amazing",
        "Excellent quality, highly recommended",
        "Great value for money",
        "Outstanding service and support",
        "Best purchase I've made",
        "This is fantastic, exceeded expectations",
        "Wonderful experience, very satisfied",
        "Top quality product",
        "I'm very happy with this",
        "Perfect, exactly what I needed",
    ]
    
    # Corresponding labels: 0=negative, 1=neutral, 2=positive
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10 negative
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 neutral
              2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # 10 positive
    
    # Convert labels to numpy array for consistency with scikit-learn
    y = np.array(labels, dtype=np.int64)
    
    # Create and fit the vectorizer
    # TfidfVectorizer converts text to numerical features
    # TF-IDF (Term Frequency-Inverse Document Frequency) gives higher weight
    # to words that are important but not too common
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limit to top 100 features for simplicity
        stop_words='english',  # Remove common English words (the, a, an, etc.)
        lowercase=True,  # Convert all text to lowercase
        ngram_range=(1, 2),  # Use both single words and 2-word phrases
    )
    
    # Transform text data into numerical features
    # fit_transform does two things:
    # 1. fit: learns the vocabulary from the training data
    # 2. transform: converts the text to numerical features
    X = vectorizer.fit_transform(texts)
    
    # Convert sparse matrix to dense numpy array for easier handling
    # When expanding to a more robust production grade with large datasets,
    # you might keep it sparse for memory efficiency
    X = X.toarray()
    
    return X, y, vectorizer


def get_data_info(X: np.ndarray, y: np.ndarray) -> None:
    """
    Print information about the loaded dataset.
    
    This is a utility function to help understand the data structure.
    Useful for debugging and understanding what we're working with.
    
    Args:
        X: Feature matrix
        y: Target labels
    """
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]}: {count}")

