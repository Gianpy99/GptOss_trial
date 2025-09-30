#!/bin/bash

# Evaluate the trained model using the specified test dataset

# Load configuration
CONFIG_FILE="../configs/training.yaml"
LORA_CONFIG="../configs/lora.yaml"
DATASET_CONFIG="../configs/dataset.yaml"

# Set up the environment (if needed)
# source /path/to/your/venv/bin/activate

# Run the evaluation script
python ../src/cli.py evaluate --config $CONFIG_FILE --lora-config $LORA_CONFIG --dataset-config $DATASET_CONFIG