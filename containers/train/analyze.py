#!/usr/bin/env python3
"""
Simple script to analyze results locally
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run
from AnalyzeResults import main

if __name__ == "__main__":
    print("Analyzing Hyperparameter Tuning Results...")
    print("=" * 80)
    main()
