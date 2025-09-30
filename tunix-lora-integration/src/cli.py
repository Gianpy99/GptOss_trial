import argparse
from src.lora.training import train_model
from src.lora.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora.yaml",
        help="Path to the configuration file for LoRA fine-tuning"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Flag to start the training process"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Flag to evaluate the trained model"
    )

    args = parser.parse_args()

    if args.train:
        config = load_config(args.config)
        train_model(config)
    elif args.evaluate:
        print("Evaluation functionality is not yet implemented.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()