import json
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def normalize_data(data):
    data_array = np.array(data)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    normalized_data = (data_array - mean) / std
    return normalized_data

def tokenize_data(data, tokenizer):
    tokenized_data = [tokenizer.encode(text) for text in data]
    return tokenized_data

def preprocess_data(file_path, tokenizer):
    data = load_data(file_path)
    normalized_data = normalize_data(data)
    tokenized_data = tokenize_data(normalized_data, tokenizer)
    return tokenized_data

def save_preprocessed_data(tokenized_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(tokenized_data, f)