import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from .adapters import LoRAAdapter
from ..data.dataset import CustomDataset
from ..utils import load_config

class LoRATrainer:
    def __init__(self, model, config_path):
        self.config = load_config(config_path)
        self.model = LoRAAdapter(model, self.config['lora'])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])
        self.train_loader = DataLoader(CustomDataset(self.config['dataset']['train_data']), 
                                       batch_size=self.config['training']['batch_size'], 
                                       shuffle=True)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.train_loader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(self.train_loader):.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()