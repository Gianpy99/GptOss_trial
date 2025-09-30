def load_lora_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_lora_config(config, config_path):
    import yaml
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def preprocess_data(data):
    # Implement any necessary preprocessing steps for LoRA here
    return data

def load_dataset(dataset_path):
    import pandas as pd
    return pd.read_csv(dataset_path)

def get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'