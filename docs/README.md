# Documentation

This directory contains all non-executable documentation for the MLOps project.

## Purpose

This project serves as a learning resource and portfolio piece demonstrating production-grade MLOps practices. The goal is to showcase:

1. **Clean, maintainable code** with explicit patterns over shorthand
2. **Industry-standard tooling** (GitHub Actions, MLflow, FastAPI)
3. **Complete MLOps lifecycle** from training to production monitoring
4. **Best practices** for Python development in ML contexts

## Documentation Files

- **[architecture.md](architecture.md)** - System architecture, components, and how they interact
- **[mlops-stages.md](mlops-stages.md)** - Detailed breakdown of each MLOps stage
- **[data-versioning.md](data-versioning.md)** - Data management and versioning strategies
- **[decisions.md](decisions.md)** - Design decisions and rationale

## Project Philosophy

### Code Style

- **Explicit over implicit**: Code should be readable and self-documenting
- **Type hints**: All functions include type annotations for clarity
- **Docstrings**: Comprehensive documentation for all modules and functions
- **Error handling**: Explicit error handling with meaningful messages

### Learning Focus

This project is designed for learning, so:
- Code includes explanatory comments where patterns might be unfamiliar
- Each commit represents a logical step in building the MLOps pipeline
- Documentation explains not just "what" but "why"

### Technology Choices

- **Python 3.10**: Stable, widely supported in ML ecosystem
- **scikit-learn**: Simple, well-documented ML library for learning
- **MLflow**: Industry-standard model registry and experiment tracking
- **FastAPI**: Modern, fast API framework with automatic documentation
- **GitHub Actions**: Native CI/CD for GitHub-hosted projects
- **Docker**: Containerization for consistent environments
