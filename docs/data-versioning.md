# Data Versioning

This document describes the data versioning strategy for the MLOps project.

## Approach

For this learning project, we'll use a simplified data versioning approach:

1. **Training Data**: Stored in `data/` directory (not committed to git)
2. **Data Loading**: Scripts in `src/data/` handle data loading
3. **Data Preprocessing**: Preprocessing steps are versioned as code
4. **Data Validation**: Basic validation to ensure data quality

## Future Enhancements

As the project evolves, we may add:
- DVC (Data Version Control) for data versioning
- Data validation schemas
- Data lineage tracking
- Automated data quality checks

