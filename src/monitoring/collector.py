"""
Monitoring Data Collector

This module collects telemetry data from model predictions for monitoring
and drift detection purposes.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class PredictionCollector:
    """
    Collects prediction data for monitoring and drift detection.
    
    This class tracks predictions made by the model, storing metadata
    that can be used for:
    - Performance monitoring
    - Data drift detection
    - Model performance tracking
    - Prediction logging
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the prediction collector.
        
        Args:
            log_dir: Directory where prediction logs will be stored.
                    If None, uses logs/ directory in project root.
        """
        if log_dir is None:
            log_dir = project_root / "logs"
        
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(
        self,
        text: str,
        prediction: str,
        confidence: float,
        model_version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a prediction for monitoring purposes.
        
        This stores prediction data that can be analyzed later for:
        - Performance tracking
        - Data drift detection
        - Model behavior analysis
        
        Args:
            text: Input text that was predicted
            prediction: Predicted sentiment (negative, neutral, positive)
            confidence: Confidence score (0.0 to 1.0)
            model_version: Version of the model used (optional)
            metadata: Additional metadata to store (optional)
        """
        # Create prediction record
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "prediction": prediction,
            "confidence": float(confidence),
            "model_version": model_version,
            "metadata": metadata or {}
        }
        
        # Append to log file (one JSON object per line for easy processing)
        log_file = self.log_dir / "predictions.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve recent predictions from the log.
        
        This is useful for analyzing recent model behavior or
        checking prediction patterns.
        
        Args:
            limit: Maximum number of predictions to retrieve
        
        Returns:
            List of prediction records (most recent first)
        """
        log_file = self.log_dir / "predictions.jsonl"
        
        if not log_file.exists():
            return []
        
        predictions = []
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                # Get last N lines (most recent)
                for line in lines[-limit:]:
                    predictions.append(json.loads(line.strip()))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return predictions
    
    def get_prediction_stats(self) -> Dict:
        """
        Get statistics about logged predictions.
        
        This provides summary statistics that can be used for
        monitoring and alerting.
        
        Returns:
            Dictionary with prediction statistics
        """
        predictions = self.get_recent_predictions(limit=10000)
        
        if not predictions:
            return {
                "total_predictions": 0,
                "by_sentiment": {},
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }
        
        # Count by sentiment
        sentiment_counts = {}
        confidences = []
        
        for pred in predictions:
            sentiment = pred["prediction"]
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidences.append(pred["confidence"])
        
        return {
            "total_predictions": len(predictions),
            "by_sentiment": sentiment_counts,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0
        }


def detect_drift(predictions: List[Dict], baseline_stats: Optional[Dict] = None) -> Dict:
    """
    Detect potential data drift in predictions.
    
    This is a placeholder function for drift detection.
    When expanding to a more robust production grade, this would:
    - Compare current prediction distribution to baseline
    - Detect statistical shifts in input data
    - Alert on significant changes
    - Use techniques like PSI (Population Stability Index), KL divergence, etc.
    
    Args:
        predictions: List of recent predictions to analyze
        baseline_stats: Baseline statistics for comparison (optional)
    
    Returns:
        Dictionary with drift detection results
    """
    # Placeholder implementation
    # In production, this would perform actual statistical analysis
    
    if not predictions:
        return {
            "drift_detected": False,
            "message": "No predictions to analyze",
            "confidence": 0.0
        }
    
    # Simple placeholder: check if confidence is unusually low
    avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)
    
    drift_detected = avg_confidence < 0.5  # Simple threshold
    
    return {
        "drift_detected": drift_detected,
        "message": "Low average confidence detected" if drift_detected else "No drift detected",
        "avg_confidence": avg_confidence,
        "threshold": 0.5,
        "note": "This is a placeholder. Implement proper drift detection for production."
    }

