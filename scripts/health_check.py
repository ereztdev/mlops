#!/usr/bin/env python3
"""
Health Check Script

This script verifies that the project setup is correct:
- All required directories exist
- Python dependencies can be imported
- Basic structure is in place
"""

import sys
from pathlib import Path


def check_directory_structure() -> bool:
    """Verify that all required directories exist."""
    print("Checking directory structure...")
    
    # Get the project root directory (parent of scripts/)
    # This script is in scripts/, so go up one level to find project root
    # This ensures the check works regardless of where the script is run from
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Only check directories needed for application runtime
    # Note: docs/ is excluded from Docker builds (not needed in container)
    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "tests",
        # "docs",  # Excluded - not needed in container, only in repo
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        # Check relative to project root
        # This works whether run from project root, scripts/, or inside Docker
        path = project_root / dir_path
        if path.exists() and path.is_dir():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def check_imports() -> bool:
    """Verify that required Python packages can be imported."""
    print("\nChecking Python imports...")
    
    # Core ML libraries
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("mlflow", "mlflow"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pytest", "pytest"),
    ]
    
    all_imported = True
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (not installed)")
            all_imported = False
    
    return all_imported


def check_python_version() -> bool:
    """Verify Python version is 3.10 or higher."""
    print("\nChecking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def main() -> int:
    """Run all health checks."""
    print("=" * 50)
    print("MLOps Project Health Check")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Python Version", check_python_version),
        ("Python Imports", check_imports),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! Project setup is correct.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

