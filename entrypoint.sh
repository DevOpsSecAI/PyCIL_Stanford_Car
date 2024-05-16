#!/bin/bash
set -e

# Ensure the training script is executable
chmod +x ./train.sh

# Log the start of the training process
echo "Starting training process..."

# Execute the training script with the specified configuration
exec ./train.sh ./exps/simplecil.json
