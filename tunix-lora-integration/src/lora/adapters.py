from typing import Any, Dict, List
import torch
from torch import nn

class LoRAAdapter:
    def __init__(self, model: nn.Module, rank: int):
        self.model = model
        self.rank = rank
        self.lora_layers = self._create_lora_layers()

    def _create_lora_layers(self) -> Dict[str, nn.Module]:
        lora_layers = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                lora_layer = nn.Linear(param.size(1), param.size(0), bias=False)
                nn.init.kaiming_uniform_(lora_layer.weight, a=math.sqrt(5))
                lora_layers[name] = lora_layer
        return lora_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, lora_layer in self.lora_layers.items():
            if name in self.model.state_dict():
                original_weight = self.model.state_dict()[name]
                lora_output = lora_layer(x)
                x = x + lora_output @ original_weight.T
        return x

    def get_parameters(self) -> List[Dict[str, Any]]:
        return [{'params': layer.parameters(), 'lr': 1e-4} for layer in self.lora_layers.values()]

    def apply_lora(self):
        for name, lora_layer in self.lora_layers.items():
            if name in self.model.state_dict():
                original_weight = self.model.state_dict()[name]
                self.model.state_dict()[name].data += lora_layer.weight.data
                lora_layer.weight.requires_grad = False

def load_lora_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config