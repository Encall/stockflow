#!/usr/bin/env python3
"""
Simple script to run hyperparameter tuning locally
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run
from src.Hyperparameter.HyperparameterTuning import main

if __name__ == "__main__":
    print("Starting Hyperparameter Tuning...")
    print("=" * 80)
    main()
