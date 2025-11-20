#!/bin/bash
# Test Docker build

echo "Testing Docker build..."
docker build -t mlops:test .

if [ $? -eq 0 ]; then
    echo "✓ Docker build successful!"
    echo ""
    echo "Testing health check inside container..."
    docker run --rm mlops:test python scripts/health_check.py
else
    echo "✗ Docker build failed!"
    exit 1
fi

