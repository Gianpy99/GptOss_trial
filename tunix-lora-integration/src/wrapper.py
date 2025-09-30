from lora.adapters import LoRAAdapter
from lora.training import LoRATrainer
from data.dataset import Dataset
from data.preprocess import preprocess_data
from models.model import Model
import yaml

class OllamaWrapper:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.model = Model(self.config['model'])
        self.dataset = Dataset(self.config['dataset'])
        self.lora_adapter = LoRAAdapter(self.config['lora'])
        self.trainer = LoRATrainer(self.model, self.dataset, self.lora_adapter)

    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def fine_tune(self):
        preprocessed_data = preprocess_data(self.dataset)
        self.trainer.train(preprocessed_data)

    def evaluate(self):
        return self.trainer.evaluate()

if __name__ == "__main__":
    wrapper = OllamaWrapper('configs/training.yaml')
    wrapper.fine_tune()
    results = wrapper.evaluate()
    print(results)