# FlowForge 
### an MLOps Project

A production-ready MLOps pipeline demonstrating best practices for machine learning operations, from training to deployment and monitoring.

## Overview

This project implements a complete MLOps workflow for a sentiment analysis model, showcasing:

- **Automated CI/CD** with GitHub Actions
- **Model Registry** with MLflow
- **Containerized Training** with Docker
- **REST API** for inference (FastAPI)
- **Web Interface** for interactive demos
- **Model Monitoring** and drift detection

## Project Structure

```
mlops/
├── src/
│   ├── data/          # Data processing and loading utilities
│   ├── models/         # Model definitions and architectures
│   ├── training/       # Training scripts and pipelines
│   └── inference/      # API and inference code
├── tests/              # Unit and integration tests
├── docs/               # Documentation and architecture notes
├── scripts/            # Utility scripts (health checks, etc.)
├── .github/            # GitHub Actions workflows
└── requirements.txt    # Python dependencies
```

## Quick Start

### Verify Setup (Health Check)

Before running anything, verify your setup is correct:

**Using Docker:**
```bash
# Build the image
docker build -t mlops:latest .

# Run health check
docker run --rm mlops:latest python scripts/health_check.py
```

**Or use the test script:**
```bash
chmod +x scripts/test_docker.sh
./scripts/test_docker.sh
```

**Local (if you have Python installed):**
```bash
python scripts/health_check.py
```

### Using Docker (Recommended)

Since this project is containerized, you can run everything without installing Python locally:

1. **Build the Docker image:**
   ```bash
   docker build -t mlops:latest .
   ```

2. **Run training:**
   ```bash
   docker run --rm mlops:latest python src/training/train.py
   ```

3. **Run the inference API:**
   ```bash
   docker run -p 8000:8000 mlops:latest python src/inference/app.py
   ```

### Local Development (Optional)

If you have Python 3.10+ installed locally:

1. Create a virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python src/training/train.py
   ```

4. Start the inference API:
   ```bash
   python src/inference/app.py
   ```

## Documentation

See the `/docs` directory for detailed documentation:
- [Architecture Overview](docs/architecture.md)
- [MLOps Stages](docs/mlops-stages.md)
- [Data Versioning](docs/data-versioning.md)
- [Design Decisions](docs/decisions.md)

## License

MIT
