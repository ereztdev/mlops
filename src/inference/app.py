"""
FastAPI Inference Application

This module provides a REST API for sentiment analysis predictions.
It loads a trained model and serves predictions via HTTP endpoints.
"""

import sys
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.load_model import load_model_and_vectorizer, predict_sentiment
from src.monitoring.collector import PredictionCollector

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for sentiment analysis predictions",
    version="1.0.0"
)

# Global variables to cache loaded model (loaded once at startup)
_model = None
_vectorizer = None
_collector = None


# Request/Response models using Pydantic for validation
class PredictionRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1)


class PredictionResponse(BaseModel):
    """Response model for sentiment prediction."""
    text: str = Field(..., description="Input text that was analyzed")
    sentiment: str = Field(..., description="Predicted sentiment: negative, neutral, or positive")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)", ge=0.0, le=1.0)


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


@app.on_event("startup")
async def load_model():
    """
    Load model and vectorizer when the application starts.
    
    This runs once when the server starts, not on every request.
    This is more efficient than loading the model for each prediction.
    """
    global _model, _vectorizer, _collector
    try:
        print("Loading model and vectorizer...")
        _model, _vectorizer = load_model_and_vectorizer()
        print("Model loaded successfully!")
        
        # Initialize prediction collector for monitoring
        _collector = PredictionCollector()
        print("Monitoring collector initialized!")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please train a model first: python src/training/train.py")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint that serves a simple web interface.
    
    This provides a basic HTML page for interactive sentiment analysis.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            textarea {
                width: 100%;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                min-height: 100px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
                width: 100%;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .negative {
                background-color: #ffebee;
                border-left: 4px solid #f44336;
            }
            .neutral {
                background-color: #fff3e0;
                border-left: 4px solid #ff9800;
            }
            .positive {
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
            }
            .sentiment-label {
                font-weight: bold;
                font-size: 18px;
                text-transform: capitalize;
            }
            .confidence {
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }
            .api-info {
                margin-top: 30px;
                padding: 15px;
                background-color: #e3f2fd;
                border-radius: 5px;
            }
            .api-info h3 {
                margin-top: 0;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Sentiment Analysis Demo</h1>
            <p>Enter text below to analyze its sentiment:</p>
            
            <textarea id="textInput" placeholder="Enter your text here..."></textarea>
            <button onclick="analyzeSentiment()">Analyze Sentiment</button>
            
            <div id="result" class="result">
                <div class="sentiment-label" id="sentiment"></div>
                <div class="confidence" id="confidence"></div>
            </div>
            
            <div class="api-info">
                <h3>API Endpoints</h3>
                <p><strong>POST</strong> <code>/predict</code> - Single prediction</p>
                <p><strong>POST</strong> <code>/predict/batch</code> - Batch predictions</p>
                <p><strong>GET</strong> <code>/docs</code> - Interactive API documentation</p>
                <p><strong>GET</strong> <code>/health</code> - Health check</p>
            </div>
        </div>
        
        <script>
            async function analyzeSentiment() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) {
                    alert('Please enter some text to analyze');
                    return;
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Prediction failed');
                    }
                    
                    const data = await response.json();
                    const resultDiv = document.getElementById('result');
                    const sentimentDiv = document.getElementById('sentiment');
                    const confidenceDiv = document.getElementById('confidence');
                    
                    // Remove previous sentiment classes
                    resultDiv.className = 'result ' + data.sentiment;
                    
                    // Update content
                    sentimentDiv.textContent = `Sentiment: ${data.sentiment}`;
                    confidenceDiv.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                    
                    // Show result
                    resultDiv.style.display = 'block';
                } catch (error) {
                    alert('Error analyzing sentiment: ' + error.message);
                }
            }
            
            // Allow Enter key to submit (Ctrl+Enter or Cmd+Enter)
            document.getElementById('textInput').addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    analyzeSentiment();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and whether the model is loaded.
    """
    if _model is None or _vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get monitoring stats if collector is available
    monitoring_stats = None
    if _collector is not None:
        monitoring_stats = _collector.get_prediction_stats()
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "API is ready to serve predictions",
        "monitoring": {
            "enabled": _collector is not None,
            "stats": monitoring_stats
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sentiment for a single text.
    
    This endpoint takes a text string and returns the predicted sentiment
    along with a confidence score.
    
    Example request:
    ```json
    {
        "text": "I love this product!"
    }
    ```
    
    Example response:
    ```json
    {
        "text": "I love this product!",
        "sentiment": "positive",
        "confidence": 0.95
    }
    ```
    """
    if _model is None or _vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        sentiment, confidence = predict_sentiment(request.text, _model, _vectorizer)
        
        # Log prediction for monitoring
        if _collector is not None:
            _collector.log_prediction(
                text=request.text,
                prediction=sentiment,
                confidence=confidence,
                model_version="latest"  # Could be retrieved from MLflow or config
            )
        
        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts in a single request.
    
    This endpoint takes a list of texts and returns predictions for all of them.
    More efficient than making multiple single prediction requests.
    
    Example request:
    ```json
    {
        "texts": [
            "I love this product!",
            "This is terrible",
            "It's okay, nothing special"
        ]
    }
    ```
    """
    if _model is None or _vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        predictions = []
        for text in request.texts:
            sentiment, confidence = predict_sentiment(text, _model, _vectorizer)
            
            # Log each prediction for monitoring
            if _collector is not None:
                _collector.log_prediction(
                    text=text,
                    prediction=sentiment,
                    confidence=confidence,
                    model_version="latest"
                )
            
            predictions.append(
                PredictionResponse(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence
                )
            )
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

