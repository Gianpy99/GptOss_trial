#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set environment variables
export CONFIG_PATH=configs/training.yaml
export LORA_CONFIG_PATH=configs/lora.yaml
export DATASET_CONFIG_PATH=configs/dataset.yaml

# Run the training script
python src/lora/training.py --config $CONFIG_PATH --lora-config $LORA_CONFIG_PATH --dataset-config $DATASET_CONFIG_PATH

# Deactivate the virtual environment if it was activated
# deactivate