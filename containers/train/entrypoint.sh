#!/bin/bash
set -e

# Entrypoint script for training container
# Passes all arguments directly to the main.py script

# Execute main.py with all passed arguments
exec python src/Hyperparameter/main.py "$@"
