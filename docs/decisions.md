# Design Decisions

This document records key design decisions and their rationale.

## Decision Log

### 2024: Project Initialization

**Decision**: Use Python 3.10
- **Rationale**: Stable, widely supported in ML ecosystem, good balance of features and compatibility
- **Alternatives considered**: Python 3.11 (newer, but some ML libraries lag), Python 3.9 (older)

**Decision**: Use scikit-learn for ML
- **Rationale**: Simple, well-documented, perfect for learning MLOps without getting lost in ML complexity
- **Alternatives considered**: TensorFlow/PyTorch (overkill for learning MLOps), XGBoost (more complex)

**Decision**: Use MLflow for model registry
- **Rationale**: Industry standard, open-source, integrates well with scikit-learn
- **Alternatives considered**: Weights & Biases (more features but proprietary), custom solution (too much work)

**Decision**: Use FastAPI for inference API
- **Rationale**: Modern, fast, automatic OpenAPI docs, easy to learn
- **Alternatives considered**: Flask (older, less features), Django (overkill for API-only)

**Decision**: Use GitHub Actions for CI/CD
- **Rationale**: Native to GitHub, free for public repos, widely used
- **Alternatives considered**: GitLab CI (user wanted GitHub showcase), Jenkins (too complex)

**Decision**: Containerize from the start
- **Rationale**: User doesn't have local Python stack, ensures consistent environments
- **Alternatives considered**: Local development only (doesn't work for user's setup)

